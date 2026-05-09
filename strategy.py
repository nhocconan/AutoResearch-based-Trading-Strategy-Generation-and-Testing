# 6H_Weekly_Camarilla_R3S3_Breakout_Trend_Filter
# Hypothesis: On 6-hour chart, use weekly Camarilla R3/S3 levels as key support/resistance.
# Breakout with confirmation from 1-week EMA trend and volume spike.
# Works in bull markets via breakouts, in bear markets via breakdowns and trend-following.
# Targets 12-37 trades per year (50-150 total over 4 years) with discrete sizing 0.25.
# Focus on BTC/ETH; avoids overtrading via strict multi-factor entry.

#!/usr/bin/env python3
name = "6H_Weekly_Camarilla_R3S3_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels and EMA trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 12:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot point and Camarilla levels (R3, S3)
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r3_w = pivot_w + (range_w * 1.1 / 4)
    s3_w = pivot_w - (range_w * 1.1 / 4)
    
    # Weekly EMA12 for trend filter
    ema12_w = pd.Series(close_w).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align to 6h
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    ema12_w_aligned = align_htf_to_ltf(prices, df_w, ema12_w)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4-day avg)
    volume_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i]) or np.isnan(ema12_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + above weekly EMA12 + volume confirmation
            if close[i] > r3_w_aligned[i] and close[i] > ema12_w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + below weekly EMA12 + volume confirmation
            elif close[i] < s3_w_aligned[i] and close[i] < ema12_w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly EMA12 (trend change)
            if close[i] < ema12_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly EMA12 (trend change)
            if close[i] > ema12_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals