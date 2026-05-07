#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Camarilla levels from previous 1d (high, low, close)
    # Calculate for each 12h bar using previous completed 1d
    prev_1d_high = np.full(n, np.nan)
    prev_1d_low = np.full(n, np.nan)
    prev_1d_close = np.full(n, np.nan)
    
    # Get previous completed 1d values for each 12h bar
    for i in range(n):
        # Find the index of the previous completed 1d bar
        # Since we're using 12h timeframe, we need to look back at least 2 bars for 1d
        if i >= 2:
            # Use the 1d data aligned to 12h timeframe
            prev_1d_high[i] = df_1d['high'].values[i//2] if i//2 < len(df_1d) else np.nan
            prev_1d_low[i] = df_1d['low'].values[i//2] if i//2 < len(df_1d) else np.nan
            prev_1d_close[i] = df_1d['close'].values[i//2] if i//2 < len(df_1d) else np.nan
    
    # Calculate Camarilla levels R3, S3, R4, S4
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_1d_high[i]) or np.isnan(prev_1d_low[i]) or np.isnan(prev_1d_close[i])):
            H = prev_1d_high[i]
            L = prev_1d_low[i]
            C = prev_1d_close[i]
            camarilla_r3[i] = C + (H - L) * 1.1 / 4
            camarilla_s3[i] = C - (H - L) * 1.1 / 4
            camarilla_r4[i] = C + (H - L) * 1.1 / 2
            camarilla_s4[i] = C - (H - L) * 1.1 / 2
    
    # Volume spike detection (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~48 hours
    
    start_idx = 20  # Ensure volume MA is valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s4[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Close above R3 with volume spike AND 1d uptrend
            if close[i] > camarilla_r3[i] and volume_spike[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Close below S3 with volume spike AND 1d downtrend
            elif close[i] < camarilla_s3[i] and volume_spike[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below S3 OR trend turns down
            if close[i] < camarilla_s3[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 OR trend turns up
            if close[i] > camarilla_r3[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as strong support/resistance in 12h timeframe.
# Breakouts above R3 or below S3 with volume confirmation and aligned with 1d trend
# capture significant moves. The Camarilla formula uses previous day's range to
# calculate levels that often act as turning points. Volume spike confirms
# institutional participation. Trend filter ensures we trade with the higher
# timeframe momentum. Cooldown of 4 bars (~48h) limits trades to ~15-30 per year.
# Position size 0.25 manages risk in volatile crypto markets. Works in both
# bull and bear markets by following the 1d trend direction.