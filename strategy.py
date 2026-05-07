#!/usr/bin/env python3
name = "4h_1d_Camarilla_R3S3_Reversion_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Daily close for trend filter
    daily_close = df_1d['close'].values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily Camarilla pivot levels from previous day (R3/S3 levels)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla R3 and S3 levels (stronger reversal levels)
    s3 = prev_close - (range_hl * 1.125)
    r3 = prev_close + (range_hl * 1.125)
    
    # Align daily levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume spike detection: 8-period average
    vol_ma_8 = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 8)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(vol_ma_8[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion long: price touches S3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_8[i] * 2.0
            daily_uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] <= s3_aligned[i] and vol_condition and daily_uptrend:
                signals[i] = 0.30
                position = 1
            # Mean reversion short: price touches R3 with volume and daily downtrend
            elif close[i] >= r3_aligned[i] and vol_condition and not daily_uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns to daily pivot or volume drops significantly
            pivot_val = (df_1d['high'].shift(1).iloc[i//6] + df_1d['low'].shift(1).iloc[i//6] + df_1d['close'].shift(1).iloc[i//6]) / 3 if i//6 < len(df_1d) and not (np.isnan(df_1d['high'].shift(1).iloc[i//6]) or np.isnan(df_1d['low'].shift(1).iloc[i//6]) or np.isnan(df_1d['close'].shift(1).iloc[i//6])) else np.nan
            if not np.isnan(pivot_val):
                pivot_aligned = align_htf_to_ltf(prices, df_1d, np.array([pivot_val] * len(df_1d)))[i]
                if close[i] >= pivot_aligned or volume[i] < vol_ma_8[i] * 1.1:
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns to daily pivot or volume drops significantly
            pivot_val = (df_1d['high'].shift(1).iloc[i//6] + df_1d['low'].shift(1).iloc[i//6] + df_1d['close'].shift(1).iloc[i//6]) / 3 if i//6 < len(df_1d) and not (np.isnan(df_1d['high'].shift(1).iloc[i//6]) or np.isnan(df_1d['low'].shift(1).iloc[i//6]) or np.isnan(df_1d['close'].shift(1).iloc[i//6])) else np.nan
            if not np.isnan(pivot_val):
                pivot_aligned = align_htf_to_ltf(prices, df_1d, np.array([pivot_val] * len(df_1d)))[i]
                if close[i] <= pivot_aligned or volume[i] < vol_ma_8[i] * 1.1:
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Mean reversion at strong Camarilla R3/S3 levels with daily trend filter
# - Uses stronger R3/S3 levels (1.125x range) for more significant reversal zones
# - Long when price touches S3 with volume spike in daily uptrend (buy the dip in uptrend)
# - Short when price touches R3 with volume spike in daily downtrend (sell the rally in downtrend)
# - Volume spike requirement (2.0x average) filters for institutional participation
# - Works in both bull (buy S3 dips in uptrend) and bear (sell R3 rallies in downtrend)
# - Exit when price returns to daily pivot or volume weakens
# - Position size 0.30 targets ~20-40 trades/year, avoiding fee drag
# - Focuses on BTC/ETH with stronger levels for better reliability