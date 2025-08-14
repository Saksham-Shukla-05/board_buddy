// import { prepareVectorStore } from "@/lib/prepareTextbook";
// import { askWithGroq } from "@/lib/askGroq";

// let store: any;

// export async function POST(req: Request) {
//   const { prompt } = await req.json();

//   if (!store) {
//     store = await prepareVectorStore();
//   }

//   const answer = await askWithGroq(prompt, store);
//   return Response.json({ answer });
// }
