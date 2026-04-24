#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout + 1d EMA34 trend filter + volume spike.
- Primary timeframe: 12h for execution, HTF: 1d for EMA34 trend filter.
- Camarilla levels: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2.
- Trend filter: EMA34(1d) slope > 0 = uptrend, < 0 = downtrend.
- Volume confirmation: current volume > 2.0x 20-period volume MA for strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying H3 breakouts in uptrend, in bear via selling L3 breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA34 slope (trend direction)
    ema34_slope = np.zeros_like(ema_34_1d_aligned)
    ema34_slope[1:] = ema_34_1d_aligned[1:] - ema_34_1d_aligned[:-1]
    
    # Camarilla levels from 1d (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    H3 = close_1d + 1.1 * rng / 2
    L3 = close_1d - 1.1 * rng / 2
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 buffer + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_slope[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trend filter: EMA34 slope > 0 = uptrend, < 0 = downtrend
            if ema34_slope[i] > 0:
                # Uptrend: buy on H3 breakout with volume spike
                if close[i] > H3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            elif ema34_slope[i] < 0:
                # Downtrend: sell on L3 breakdown with volume spike
                if close[i] < L3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to midpoint of H3/L3 or breaks below L3
            midpoint = (H3_aligned[i] + L3_aligned[i]) / 2
            if close[i] < midpoint or close[i] < L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint of H3/L3 or breaks above H3
            midpoint = (H3_aligned[i] + L3_aligned[i]) / 2
            if close[i] > midpoint or close[i] > H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0