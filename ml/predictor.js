/**
 * Produce Your Own Movie — ML rating predictor (pure JS, no dependencies).
 *
 * Implements LightGBM tree traversal from model_trees.json.
 * Drop this file into your Next.js API route or Vercel serverless function.
 *
 * Usage:
 *   import { predictRating } from './predictor.js'
 *   const result = predictRating({
 *     director: 'Sanjay Leela Bhansali',
 *     hero: 'Ranveer Singh',
 *     heroine: 'Deepika Padukone',
 *     villain: 'Unknown',
 *     musicComposer: 'Pritam',
 *     genres: ['Romance', 'Drama'],
 *   })
 *   // { rawScore: 6.89, displayScore: 6.89, stars: 3.44 }
 */

import modelData from './model_trees.json' assert { type: 'json' }

const GAME_MIN_DISPLAY = 5.0

const ALL_GENRES = [
  'Drama', 'Comedy', 'Romance', 'Action', 'Thriller', 'Crime',
  'Adventure', 'History', 'Family', 'Horror', 'Mystery', 'Fantasy',
  'Animation', 'Music', 'War', 'Biography',
]

const FEATURE_NAMES = modelData.feature_names  // 26 features in order
const CREW_FEATURES = modelData.cat_features    // ['director','hero','heroine','villain','music_composer']
const NAME_TO_CODE  = modelData.name_to_code    // {director: {name: code, ...}, ...}
const TREES         = modelData.trees

/**
 * Walk a single LightGBM decision tree and return the leaf value.
 * @param {object} node  - tree node from model_trees.json
 * @param {number[]} row - feature vector (same order as FEATURE_NAMES)
 */
function walkTree(node, row) {
  if (node.leaf) return node.leaf_value

  const val = row[node.feature]

  let goLeft
  if (node.type === '==') {
    // Categorical split — threshold is pipe-separated category codes, e.g. "4||5||16"
    const codes = node.threshold.split('||').map(Number)
    goLeft = codes.includes(val)
  } else {
    // Numeric split — threshold is a number, go left if val <= threshold
    goLeft = val <= node.threshold
  }

  return walkTree(goLeft ? node.left : node.right, row)
}

/**
 * Build a 26-element feature vector from game inputs.
 */
function buildFeatureVector({ director, hero, heroine, villain, musicComposer, genres }) {
  const crewInputs = {
    director:        director      || 'Unknown',
    hero:            hero          || 'Unknown',
    heroine:         heroine       || 'Unknown',
    villain:         villain       || 'Unknown',
    music_composer:  musicComposer || 'Unknown',
  }

  // Map crew names → category codes (unseen names → code for 'Unknown')
  const crewCodes = {}
  for (const feat of CREW_FEATURES) {
    const lookup   = NAME_TO_CODE[feat]
    const name     = crewInputs[feat]
    const unknown  = lookup['Unknown'] ?? 0
    crewCodes[feat] = lookup[name] !== undefined ? lookup[name] : unknown
  }

  // Genre binary flags
  const genreSet = new Set(genres || [])
  const genreFlags = {}
  for (const g of ALL_GENRES) {
    genreFlags[`genre_${g.toLowerCase()}`] = genreSet.has(g) ? 1 : 0
  }

  // Curated flags — names that are in the game's curated lists
  const CURATED = {
    director:       new Set(Object.keys(NAME_TO_CODE.director).filter(n =>
                      CURATED_DIRECTORS.has(n))),
    hero:           new Set(Object.keys(NAME_TO_CODE.hero).filter(n =>
                      CURATED_HEROES.has(n))),
    heroine:        new Set(Object.keys(NAME_TO_CODE.heroine).filter(n =>
                      CURATED_HEROINES.has(n))),
    villain:        new Set(Object.keys(NAME_TO_CODE.villain).filter(n =>
                      CURATED_VILLAINS.has(n))),
    music_composer: new Set(Object.keys(NAME_TO_CODE.music_composer).filter(n =>
                      CURATED_COMPOSERS.has(n))),
  }

  const is_curated_director  = CURATED_DIRECTORS.has(crewInputs.director)       ? 1 : 0
  const is_curated_hero      = CURATED_HEROES.has(crewInputs.hero)              ? 1 : 0
  const is_curated_heroine   = CURATED_HEROINES.has(crewInputs.heroine)         ? 1 : 0
  const is_curated_villain   = CURATED_VILLAINS.has(crewInputs.villain)         ? 1 : 0
  const is_curated_composer  = CURATED_COMPOSERS.has(crewInputs.music_composer) ? 1 : 0

  // Build vector in exact FEATURE_NAMES order
  return FEATURE_NAMES.map(name => {
    if (CREW_FEATURES.includes(name)) return crewCodes[name]
    if (name in genreFlags)           return genreFlags[name]
    if (name === 'is_curated_director') return is_curated_director
    if (name === 'is_curated_hero')     return is_curated_hero
    if (name === 'is_curated_heroine')  return is_curated_heroine
    if (name === 'is_curated_villain')  return is_curated_villain
    if (name === 'is_curated_composer') return is_curated_composer
    return 0
  })
}

/**
 * Predict movie rating from crew + genre inputs.
 *
 * @param {object} inputs
 * @param {string} inputs.director
 * @param {string} inputs.hero
 * @param {string} inputs.heroine
 * @param {string} inputs.villain
 * @param {string} inputs.musicComposer
 * @param {string[]} inputs.genres
 * @returns {{ rawScore: number, displayScore: number, stars: number }}
 */
export function predictRating(inputs) {
  const row = buildFeatureVector(inputs)
  const raw = TREES.reduce((sum, tree) => sum + walkTree(tree, row), 0)
  const clamped = Math.max(0, Math.min(10, raw))
  const display = Math.max(clamped, GAME_MIN_DISPLAY)
  return {
    rawScore:     Math.round(clamped * 100) / 100,
    displayScore: Math.round(display * 100) / 100,
    stars:        Math.round((display / 2) * 100) / 100,
  }
}

// ── Curated lists (copied from feature_template.py) ──────────────────────────
const CURATED_DIRECTORS = new Set([
  'Yash Chopra','Sanjay Leela Bhansali','Rajkumar Hirani','Karan Johar',
  'Aditya Chopra','Rohit Shetty','Farah Khan','Anurag Basu','Imtiaz Ali',
  'Zoya Akhtar','Ashutosh Gowariker','Sandeep Reddy Vanga','Ayan Mukerji',
  'Anurag Kashyap','Amar Kaushik','Atlee','Aditya Dhar','S. S. Rajamouli',
])
const CURATED_HEROES = new Set([
  'Ranbir Kapoor','Ranveer Singh','Vicky Kaushal','Shahid Kapoor',
  'Kartik Aaryan','Varun Dhawan','Ayushmann Khurrana','Sidharth Malhotra',
  'Emraan Hashmi','R. Madhavan','Riteish Deshmukh','Sushant Singh Rajput',
  'Shah Rukh Khan','Salman Khan','Aamir Khan','Akshay Kumar','Ajay Devgn',
  'Hrithik Roshan','Sanjay Dutt','Saif Ali Khan','Prabhas','Allu Arjun',
  'Ram Charan','Jr. NTR','Mahesh Babu',
])
const CURATED_HEROINES = new Set([
  'Deepika Padukone','Alia Bhatt','Kareena Kapoor Khan','Katrina Kaif',
  'Priyanka Chopra Jonas','Shraddha Kapoor','Kiara Advani','Kriti Sanon',
  'Rashmika Mandanna','Triptii Dimri','Madhuri Dixit','Kajol',
  'Aishwarya Rai Bachchan','Yami Gautam','Mrunal Thakur','Sharvari',
  'Disha Patani','Rani Mukerji','Preity Zinta','Karisma Kapoor',
  'Anushka Sharma','Janhvi Kapoor','Sara Ali Khan','Kangana Ranaut','Ananya Panday',
])
const CURATED_VILLAINS = new Set([
  'Amrish Puri','Arjun Kapoor','Rana Daggubati','Vidya Balan','Tabu',
  'John Abraham','Saif Ali Khan','Prakash Raj','Urmila Matondkar',
  'Akshaye Khanna','Vijay Sethupathi','Pankaj Tripathi','Bobby Deol',
  'Arjun Rampal','Raghav Juyal','Richa Chadha','Ranjeet','Nawazuddin Siddiqui',
  'Sonu Sood','Danny Denzongpa','Jackie Shroff','Sushmita Sen',
])
const CURATED_COMPOSERS = new Set([
  'A. R. Rahman','Pritam','Shankar-Ehsaan-Loy','Vishal-Shekhar',
  'Amit Trivedi','Anu Malik','Jatin-Lalit','Bappi Lahiri','Rajesh Roshan',
  'Sachin-Jigar','Ajay-Atul','Amaal Mallik','Tanishk Bagchi','Mithoon',
  'Salim-Sulaiman','Shantanu Moitra','Ram Sampath','Jeet Gannguli',
  'Sachet-Parampara','Darshan Raval','Himesh Reshammiya','Jasleen Royal',
])
