#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6-hour price action with weekly pivot structure and 1d volume confirmation
    # Target: 15-25 trades/year per symbol by fading at weekly R3/S3 and breaking through R4/S4
    # Weekly pivots define institutional support/resistance; volume confirms breakout legitimacy
    # Works in bull/bear via mean-reversion at extremes and trend-following on breaks
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pp = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    r2 = pp + (high_1w - low_1w)
    s2 = pp - (high_1w - low_1w)
    r3 = high_1w + 2 * (pp - low_1w)
    s3 = low_1w - 2 * (high_1w - pp)
    r4 = r3 + (high_1w - low_1w)
    s4 = s3 - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade at weekly R3/S3: sell at resistance, buy at support
            if close[i] >= r3_aligned[i] and volume[i] < vol_ma20_1d_aligned[i]:
                signals[i] = -0.25  # Short at R3 with low volume (weakness)
                position = -1
            elif close[i] <= s3_aligned[i] and volume[i] < vol_ma20_1d_aligned[i]:
                signals[i] = 0.25   # Long at S3 with low volume (weakness)
                position = 1
            # Breakout continuation at R4/S4: break with volume
            elif close[i] > r4_aligned[i] and volume[i] > 2.0 * vol_ma20_1d_aligned[i]:
                signals[i] = 0.25   # Long breakout with volume
                position = 1
            elif close[i] < s4_aligned[i] and volume[i] > 2.0 * vol_ma20_1d_aligned[i]:
                signals[i] = -0.25  # Short breakdown with volume
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: return to S3 or breakdown below S4
                if close[i] <= s3_aligned[i] or close[i] < s4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: return to R3 or break above R4
                if close[i] >= r3_aligned[i] or close[i] > r4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_R3S3_R4S4_Volume_FadeBreakout_v1"
timeframe = "6h"
leverage = 1.0