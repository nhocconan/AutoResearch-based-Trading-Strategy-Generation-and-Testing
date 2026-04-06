#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13931_6d_1d_parabolic_sar_slope_v1"
timeframe = "6h"
leverage = 1.0

# Hypothesis: Parabolic SAR on 1d (trend direction) + SAR slope acceleration on 6h (entry timing)
# Works in bull: 1d SAR below price (uptrend) + 6h SAR slope turning up = long
# Works in bear: 1d SAR above price (downtrend) + 6h SAR slope turning down = short
# Uses SAR as dynamic stop/reversal - tight entries, low whipsaw
# Target: 60-120 total trades over 4 years (15-30/year)

def calculate_parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Calculate Parabolic SAR"""
    n = len(high)
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    af = np.zeros(n)
    ep = np.zeros(n)  # extreme point
    
    # Initialize
    sar[0] = low[0]
    trend[0] = 1
    af[0] = af_start
    ep[0] = high[0]
    
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            if low[i] <= sar[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = low[i]
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            if high[i] >= sar[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = high[i]
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    return sar, trend

def calculate_sar_slope(sar, period=3):
    """Calculate SAR slope (rate of change)"""
    sar_series = pd.Series(sar)
    slope = sar_series.diff(period) / period  # change over period
    return slope.values

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Parabolic SAR for trend direction
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    sar_1d, trend_1d = calculate_parabolic_sar(high_1d, low_1d)
    sar_1d_aligned = align_htf_to_ltf(prices, df_1d, sar_1d)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 6h data for SAR slope and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 6h Parabolic SAR and its slope
    sar_6h, trend_6h = calculate_parabolic_sar(high, low)
    sar_slope = calculate_sar_slope(sar_6h, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(10, 3) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sar_1d_aligned[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(sar_slope[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter from 1d SAR
        # In uptrend: SAR below price, in downtrend: SAR above price
        trend_up = sar_1d_aligned[i] < close[i]
        trend_down = sar_1d_aligned[i] > close[i]
        
        # SAR slope acceleration on 6h
        # Positive slope = SAR moving up (accelerating uptrend)
        # Negative slope = SAR moving down (accelerating downtrend)
        slope_up = sar_slope[i] > 0
        slope_down = sar_slope[i] < 0
        
        # Entry signals: trend alignment + slope acceleration
        long_signal = trend_up and slope_up
        short_signal = trend_down and slope_down
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when trend turns down or slope turns negative
            if not trend_up or not slope_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when trend turns up or slope turns positive
            if not trend_down or not slope_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals