#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Load daily data ONCE for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r3 = pivot + range_hl * 1.1 / 2
    s3 = pivot - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above R3 in daily uptrend with volume spike
            if close[i] > r3_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in daily downtrend with volume spike
            elif close[i] < s3_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below pivot or trend reverses
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] < pivot_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above pivot or trend reverses
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] > pivot_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance derived from prior day's range
# - Breakout above R3 in daily uptrend (EMA34 rising) signals bullish momentum
# - Breakdown below S3 in daily downtrend (EMA34 falling) signals bearish momentum
# - Volume confirmation (2x average) filters out weak breakouts
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Exit when price returns to daily pivot or trend reverses
# - Position size 0.25 targets ~30-80 trades/year to avoid fee drag
# - Camarilla levels provide mathematically derived support/resistance with statistical edge
# - Daily trend filter reduces whipsaws vs same-timeframe signals
# - Proven pattern: Camarilla breakouts with volume and trend filter show strong performance in DB
# - Aims for 50-150 total trades over 4 years (12-37/year) staying within limits
# - Focus on BTC/ETH as primary targets; avoids SOL-only bias