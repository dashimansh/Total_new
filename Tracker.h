#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "KalmanTracker.h"

enum class ETrackerType
{
    CSRT,
    KCF,
    CAMSHIFT,
    TEMPLATE
};

enum class ETrackState
{
    Idle,
    Tracking,
    Occluded,
    Lost
};

class ObjectTracker
{
public:
    ObjectTracker();

    bool Init(
        const cv::Mat& Frame,
        const cv::Rect& BBox,
        ETrackerType Type =
            ETrackerType::CSRT);

    cv::Rect Update(
        const cv::Mat& Frame);

    bool Reinit(
        const cv::Mat& Frame,
        const cv::Rect& NewBBox);

    void Reset();
    void Draw(cv::Mat& Frame);

    ETrackState GetState() const
    {
        return State;
    }
    cv::Rect GetCurrentBox() const
    {
        return CurrentBox;
    }
    cv::Rect GetPredictedBox() const
    {
        return KFilter.GetPredictedBox();
    }
    int GetLostCount() const
    {
        return LostFrameCount;
    }
    bool IsTracking() const
    {
        return State == ETrackState::Tracking;
    }
    bool IsOccluded() const
    {
        return State == ETrackState::Occluded;
    }
    bool IsLost() const
    {
        return State == ETrackState::Lost;
    }

private:
    KalmanTracker KFilter;
    ETrackState   State       = ETrackState::Idle;
    ETrackerType  TrackerType = ETrackerType::CSRT;
    cv::Rect      CurrentBox;
    int           LostFrameCount = 0;

    cv::Ptr<cv::Tracker> CVTracker;

    cv::Mat  RoiHist;
    cv::Rect TrackWindow;
    cv::Mat  ObjTemplate;
    cv::Mat  ObjHistogram;

    cv::Ptr<cv::BackgroundSubtractorMOG2>
        BGSub;

    std::vector<double> Scales;

    bool UpdateCVTracker(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool UpdateCamShift(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool UpdateTemplate(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    cv::Rect AdaptScale(
        const cv::Mat& Frame,
        const cv::Rect& Box);

    double CompareHistogram(
        const cv::Mat& Frame,
        const cv::Rect& Box);

    bool FindByTemplateAndColor(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool FindByMotion(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool TryReacquire(
        const cv::Mat& Frame);
};
