#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price reversal at 1-day high/low with volume exhaustion and 1d EMA trend filter.
# In ranging markets (common in 2025 BTC/ETH), price often reverses near daily extremes.
# Uses: 1) Price touches 1d high/low (from prior day), 2) Volume < 50% of 20-period avg (exhaustion),
# 3) Price closes back inside prior day's range (rejection), 4) 1d EMA50 trend filter.
# Works in bull/bear by only taking reversals against the 1d trend (mean reversion in range).
# Target: 50-150 total trades over 4 years = 12-37/year. Size: 0.25.
name = "6h_DailyExtremeReversal_1dEMA50_VolumeExhaustion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for daily extremes and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Prior day's high, low, close
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # 1d EMA50 trend filter
    ema_1d = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume exhaustion: volume < 50% of 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_exhaust = volume < (0.5 * vol_ema20)
    
    # Align 1d data to 6h timeframe (wait for prior day to close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(prev_close_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ph = prev_high_aligned[i]
        pl = prev_low_aligned[i]
        pc = prev_close_aligned[i]
        ema = ema_1d_aligned[i]
        
        if position == 0:
            # Long reversal: price touches or exceeds prior day low, volume exhaustion,
            # closes back above prior day low, and below 1d EMA (downtrend = fade)
            touched_low = low[i] <= pl
            close_above_low = close[i] > pl
            below_ema = price < ema
            
            if touched_low and close_above_low and vol_exhaust[i] and below_ema:
                signals[i] = 0.25
                position = 1
            
            # Short reversal: price touches or exceeds prior day high, volume exhaustion,
            # closes back below prior day high, and above 1d EMA (uptrend = fade)
            touched_high = high[i] >= ph
            close_below_high = close[i] < ph
            above_ema = price > ema
            
            if touched_high and close_below_high and vol_exhaust[i] and above_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches prior day high (target) or breaks above 1d EMA (trend change)
            if high[i] >= ph or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches prior day low (target) or breaks below 1d EMA (trend change)
            if low[i] <= pl or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals