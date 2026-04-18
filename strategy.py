#!/usr/bin/env python3
"""
4h_Parabolic_SAR_Trend_Reverse
Hypothesis: Uses Parabolic SAR for trend following with a volume filter and EMA trend filter.
Enters long when SAR flips below price (bullish) with EMA21 > EMA50 and volume > 1.5x average.
Enters short when SAR flips above price (bearish) with EMA21 < EMA50 and volume > 1.5x average.
Exits when SAR flips back to the opposite side.
Designed to capture trends while avoiding whipsaws in sideways markets, with low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize SAR arrays
    sar = np.full(n, np.nan)
    trend = np.full(n, np.nan)  # 1 for uptrend, -1 for downtrend
    af = np.full(n, np.nan)
    ep = np.full(n, np.nan)  # extreme point
    
    # Initialize first values
    sar[0] = low[0]
    trend[0] = 1
    af[0] = af_start
    ep[0] = high[0]
    
    # Calculate SAR for each period
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # SAR cannot be above the low of the past two periods
            sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            
            # Check for trend reversal
            if sar[i] > low[i]:
                trend[i] = -1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]
                af[i] = af_start
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
            # SAR cannot be below the high of the past two periods
            sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            
            # Check for trend reversal
            if sar[i] < high[i]:
                trend[i] = 1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # EMA trend filter (21 and 50)
    ema21 = np.full(n, np.nan)
    ema50 = np.full(n, np.nan)
    k21 = 2 / (21 + 1)
    k50 = 2 / (50 + 1)
    for i in range(50, n):
        if i == 50:
            ema21[i] = np.mean(close[i-21+1:i+1]) if i >= 21 else np.nan
            ema50[i] = np.mean(close[i-50+1:i+1])
        else:
            if not np.isnan(ema21[i-1]):
                ema21[i] = close[i] * k21 + ema21[i-1] * (1 - k21)
            if not np.isnan(ema50[i-1]):
                ema50[i] = close[i] * k50 + ema50[i-1] * (1 - k50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(sar[i]) or np.isnan(ema21[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: SAR below price (bullish) with uptrend and volume filter
            if sar[i] < close[i] and ema21[i] > ema50[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: SAR above price (bearish) with downtrend and volume filter
            elif sar[i] > close[i] and ema21[i] < ema50[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: SAR flips above price (trend reversal)
            if sar[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: SAR flips below price (trend reversal)
            if sar[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Parabolic_SAR_Trend_Reverse"
timeframe = "4h"
leverage = 1.0