import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import ReactPlayer from "react-player/file";

import "highlight.js/styles/github-dark.css";

export default function CardAPIGuide() {
  const [cropPestAnalysisGuideAPIContent, setCropPestAnalysisGuideAPIContent] =
    useState("");
  const [
    coffeeFarmersInformationAPIGuideContent,
    setCoffeeFarmersInformationAPIGuideContent,
  ] = useState("");

  const fetchCropPestAnalysisApiGuide = async () => {
    try {
      const response = await fetch("/docs/CROP_PEST_ANALYSIS_API_GUIDE.md");
      if (!response.ok) {
        throw new Error("Failed to fetch the Markdown file");
      }
      const text = await response.text();
      setCropPestAnalysisGuideAPIContent(text);
    } catch (error) {
      console.error("Error fetching the Markdown file:", error);
    }
  };

  const fetchCoffeeFarmersInformationApiGuide = async () => {
    try {
      const response = await fetch(
        "/docs/COFFEE_FARMERS_INFORMATION_API_GUIDE.md"
      );
      if (!response.ok) {
        throw new Error("Failed to fetch the Markdown file");
      }
      const text = await response.text();
      setCoffeeFarmersInformationAPIGuideContent(text);
    } catch (error) {
      console.error("Error fetching the Markdown file:", error);
    }
  };

  useEffect(() => {
    fetchCropPestAnalysisApiGuide();
    fetchCoffeeFarmersInformationApiGuide();
  }, []);

  return (
    <section id="api-guide" className="pt-14 sm:pt-20 lg:pt-[130px]">
      <div className="px-4 xl:container">
        <div className="relative mx-auto mb-12 pt-6 text-center md:mb-20 lg:pt-16">
          <span className="title"> API GUIDE </span>
          <h2 className="mx-auto mb-5 max-w-[450px] font-heading text-3xl font-semibold text-dark dark:text-white sm:text-4xl md:text-[50px] md:leading-[60px]">
            How can I use AgriSense API?
          </h2>
        </div>

        <div className="w-full px-4">
          <ReactPlayer
            url="/videos/mobile-guide.mp4"
            playing
            muted
            loop
            width="100%"
            height="auto"
            controls
          />

          <div className="markdown-container bg-gray-200 p-4 my-4">
            <Markdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              children={cropPestAnalysisGuideAPIContent}
            ></Markdown>
          </div>

          <ReactPlayer
            url="/videos/CROP_PEST_ANALYSIS_API_GUIDE.mp4"
            playing
            muted
            loop
            width="100%"
            height="auto"
            controls
          />

          {/* <div className="markdown-container bg-gray-200 p-4 my-4">
            <Markdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              children={coffeeFarmersInformationAPIGuideContent}
            ></Markdown>
          </div> */}

          {/* <ReactPlayer
            url="/videos/COFFEE_FARMERS_INFORMATION_API_GUIDE.mp4"
            playing
            muted
            loop
            width="100%"
            height="auto"
            controls
          /> */}
        </div>
      </div>
    </section>
  );
}
