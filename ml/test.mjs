import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __dir = dirname(fileURLToPath(import.meta.url))
const modelData = JSON.parse(readFileSync(join(__dir, 'model_trees.json'), 'utf8'))

const GAME_MIN = 5.0
const ALL_GENRES = ['Drama','Comedy','Romance','Action','Thriller','Crime','Adventure','History','Family','Horror','Mystery','Fantasy','Animation','Music','War','Biography']
const FEATURE_NAMES = modelData.feature_names
const CREW_FEATURES = modelData.cat_features
const NAME_TO_CODE  = modelData.name_to_code
const TREES         = modelData.trees

const CURATED_DIRECTORS = new Set(['Yash Chopra','Sanjay Leela Bhansali','Rajkumar Hirani','Karan Johar','Aditya Chopra','Rohit Shetty','Farah Khan','Anurag Basu','Imtiaz Ali','Zoya Akhtar','Ashutosh Gowariker','Sandeep Reddy Vanga','Ayan Mukerji','Anurag Kashyap','Amar Kaushik','Atlee','Aditya Dhar','S. S. Rajamouli'])
const CURATED_HEROES = new Set(['Ranbir Kapoor','Ranveer Singh','Vicky Kaushal','Shahid Kapoor','Kartik Aaryan','Varun Dhawan','Ayushmann Khurrana','Sidharth Malhotra','Emraan Hashmi','R. Madhavan','Riteish Deshmukh','Sushant Singh Rajput','Shah Rukh Khan','Salman Khan','Aamir Khan','Akshay Kumar','Ajay Devgn','Hrithik Roshan','Sanjay Dutt','Saif Ali Khan','Prabhas','Allu Arjun','Ram Charan','Jr. NTR','Mahesh Babu'])
const CURATED_HEROINES = new Set(['Deepika Padukone','Alia Bhatt','Kareena Kapoor Khan','Katrina Kaif','Priyanka Chopra Jonas','Shraddha Kapoor','Kiara Advani','Kriti Sanon','Rashmika Mandanna','Triptii Dimri','Madhuri Dixit','Kajol','Aishwarya Rai Bachchan','Yami Gautam','Mrunal Thakur','Sharvari','Disha Patani','Rani Mukerji','Preity Zinta','Karisma Kapoor','Anushka Sharma','Janhvi Kapoor','Sara Ali Khan','Kangana Ranaut','Ananya Panday'])
const CURATED_VILLAINS = new Set(['Amrish Puri','Arjun Kapoor','Rana Daggubati','Vidya Balan','Tabu','John Abraham','Saif Ali Khan','Prakash Raj','Urmila Matondkar','Akshaye Khanna','Vijay Sethupathi','Pankaj Tripathi','Bobby Deol','Arjun Rampal','Raghav Juyal','Richa Chadha','Ranjeet','Nawazuddin Siddiqui','Sonu Sood','Danny Denzongpa','Jackie Shroff','Sushmita Sen'])
const CURATED_COMPOSERS = new Set(['A. R. Rahman','Pritam','Shankar-Ehsaan-Loy','Vishal-Shekhar','Amit Trivedi','Anu Malik','Jatin-Lalit','Bappi Lahiri','Rajesh Roshan','Sachin-Jigar','Ajay-Atul','Amaal Mallik','Tanishk Bagchi','Mithoon','Salim-Sulaiman','Shantanu Moitra','Ram Sampath','Jeet Gannguli','Sachet-Parampara','Darshan Raval','Himesh Reshammiya','Jasleen Royal'])

function walkTree(node, row) {
  if (node.leaf) return node.leaf_value
  const val = row[node.feature]
  const goLeft = node.type === '=='
    ? node.threshold.split('||').map(Number).includes(val)
    : val <= node.threshold
  return walkTree(goLeft ? node.left : node.right, row)
}

function predict({ director, hero, heroine, villain, musicComposer, genres }) {
  const crew = { director: director||'Unknown', hero: hero||'Unknown', heroine: heroine||'Unknown', villain: villain||'Unknown', music_composer: musicComposer||'Unknown' }
  const codes = {}
  for (const f of CREW_FEATURES) { const m = NAME_TO_CODE[f]; codes[f] = m[crew[f]] ?? m['Unknown'] ?? 0 }
  const gset = new Set(genres||[])
  const gflags = Object.fromEntries(ALL_GENRES.map(g => [`genre_${g.toLowerCase()}`, gset.has(g)?1:0]))
  const row = FEATURE_NAMES.map(n => {
    if (CREW_FEATURES.includes(n)) return codes[n]
    if (n in gflags) return gflags[n]
    if (n==='is_curated_director')  return CURATED_DIRECTORS.has(crew.director)?1:0
    if (n==='is_curated_hero')      return CURATED_HEROES.has(crew.hero)?1:0
    if (n==='is_curated_heroine')   return CURATED_HEROINES.has(crew.heroine)?1:0
    if (n==='is_curated_villain')   return CURATED_VILLAINS.has(crew.villain)?1:0
    if (n==='is_curated_composer')  return CURATED_COMPOSERS.has(crew.music_composer)?1:0
    return 0
  })
  const raw = TREES.reduce((s,t) => s + walkTree(t, row), 0)
  const clamped = Math.max(0, Math.min(10, raw))
  const display = Math.max(clamped, GAME_MIN)
  return { rawScore: +clamped.toFixed(2), displayScore: +display.toFixed(2), stars: +(display/2).toFixed(2) }
}

// ── Test cases ────────────────────────────────────────────────────────────────

const tests = [
{
label: 'Prestige director with a proven romantic lead pairing',
input: {
director: 'Sanjay Leela Bhansali',
hero: 'Ranveer Singh',
heroine: 'Deepika Padukone',
villain: 'Jim Sarbh',
musicComposer: 'Sanjay Leela Bhansali',
genres: ['Romance', 'Drama']
},
},
{
label: 'Acclaimed director with a strong dramatic cast',
input: {
director: 'Rajkumar Hirani',
hero: 'Aamir Khan',
heroine: 'Kareena Kapoor Khan',
villain: 'Boman Irani',
musicComposer: 'Shantanu Moitra',
genres: ['Drama', 'Comedy']
},
},
{
label: 'Popular action director with an established star cast',
input: {
director: 'S. S. Rajamouli',
hero: 'Prabhas',
heroine: 'Anushka Shetty',
villain: 'Rana Daggubati',
musicComposer: 'M. M. Keeravani',
genres: ['Action', 'Drama', 'Fantasy']
},
},
{
label: 'Critically acclaimed director with a musical romance',
input: {
director: 'Imtiaz Ali',
hero: 'Shah Rukh Khan',
heroine: 'Deepika Padukone',
villain: 'Unknown',
musicComposer: 'A. R. Rahman',
genres: ['Romance', 'Drama', 'Musical']
},
},
{
label: 'Strong ensemble cast with an emotional storyline',
input: {
director: 'Zoya Akhtar',
hero: 'Ranveer Singh',
heroine: 'Priyanka Chopra',
villain: 'Unknown',
musicComposer: 'Shankar-Ehsaan-Loy',
genres: ['Drama', 'Comedy']
},
},
{
label: 'High-profile action thriller with a popular composer',
input: {
director: 'Siddharth Anand',
hero: 'Hrithik Roshan',
heroine: 'Deepika Padukone',
villain: 'John Abraham',
musicComposer: 'Vishal-Shekhar',
genres: ['Action', 'Thriller']
},
},
{
label: 'Acclaimed crime thriller with strong lead performances',
input: {
director: 'Sriram Raghavan',
hero: 'Ayushmann Khurrana',
heroine: 'Tabu',
villain: 'Manav Vij',
musicComposer: 'Amit Trivedi',
genres: ['Crime', 'Thriller', 'Mystery']
},
},
{
label: 'Epic historical drama with an established creative team',
input: {
director: 'Mani Ratnam',
hero: 'Vikram',
heroine: 'Aishwarya Rai Bachchan',
villain: 'Jayam Ravi',
musicComposer: 'A. R. Rahman',
genres: ['Historical', 'Drama', 'Action']
},
},
{
label: 'Popular director with a family-friendly comedy',
input: {
director: 'Priyadarshan',
hero: 'Akshay Kumar',
heroine: 'Vidya Balan',
villain: 'Paresh Rawal',
musicComposer: 'Pritam',
genres: ['Comedy', 'Drama']
},
},
{
label: 'Acclaimed filmmaker with a powerful crime drama',
input: {
director: 'Anurag Kashyap',
hero: 'Nawazuddin Siddiqui',
heroine: 'Huma Qureshi',
villain: 'Tigmanshu Dhulia',
musicComposer: 'Sneha Khanwalkar',
genres: ['Crime', 'Drama', 'Thriller']
},
},
]

console.log('\n=== PYOM Rating Predictor — Terminal Test ===\n')
for (const { label, input } of tests) {
  const r = predict(input)
  const bar = '★'.repeat(Math.round(r.stars)) + '☆'.repeat(5 - Math.round(r.stars))
  console.log(`${label}`)
  console.log(`  → ${r.displayScore}/10  ${bar}  (${r.stars} stars)\n`)
}
