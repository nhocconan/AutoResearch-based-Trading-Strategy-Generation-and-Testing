#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Load daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day (R3, S3 levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Weekly EMA trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 level in weekly uptrend with volume spike
            if close[i] > r3_aligned[i] and vol_ma_20[i] > 0 and volume[i] > vol_ma_20[i] * 2.0 and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 level in weekly downtrend with volume spike
            elif close[i] < s3_aligned[i] and vol_ma_20[i] > 0 and volume[i] > vol_ma_20[i] * 2.0 and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below S3 level or weekly trend turns down
            if close[i] < s3_aligned[i] or ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above R3 level or weekly trend turns up
            if close[i] > r3_aligned[i] or ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with weekly trend filter and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance derived from prior day's range
# - Breakout above R3 in weekly uptrend = long signal; breakdown below S3 in weekly downtrend = short signal
# - Weekly EMA20 trend filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation (2x average) filters out false breakouts
# - Exits when price returns to opposite Camarilla level or weekly trend changes
# - Position size 0.25 balances risk/reward while limiting trades to ~15-35/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets
# - Uses actual daily Camarilla levels (not resampled) and proper MTF alignment via mtf_data
# - Targets 60-140 total trades over 4 years (15-35/year) staying within limits
# - Avoided overtrading by requiring volume spike and trend alignment for entry
# - Camarilla levels provide defined entry/exit levels with statistical edge in ranging markets