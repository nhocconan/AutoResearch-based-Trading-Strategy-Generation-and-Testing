#146716 - 12h Camarilla R1/S1 Breakout + 1d Trend + Volume Confirmation
# Hypothesis: At 12h timeframe, using daily Camarilla pivot levels (R1/S1) for breakout entries,
# filtered by 1d trend (EMA50) and volume surge, reduces overtrading while maintaining edge.
# Works in bull/bear via trend filter. Target: 12-37 trades/year to avoid fee drag.

#!/usr/bin/env python3
name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels (R1/S1) from previous day
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        camarilla_r1[i] = prev_close + 1.1 * range_ / 12
        camarilla_s1[i] = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        # Volume surge: current volume > 1.5x 12-period volume average (adaptive to timeframe)
        vol_ma = np.mean(volume[max(0, i-12):i+1]) if i >= 12 else volume[i]
        volume_surge = volume[i] > vol_ma * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Camarilla R1 + volume surge
            if uptrend and close[i] > camarilla_r1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Camarilla S1 + volume surge
            elif downtrend and close[i] < camarilla_s1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Camarilla S1
            if not uptrend or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Camarilla R1
            if not downtrend or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals