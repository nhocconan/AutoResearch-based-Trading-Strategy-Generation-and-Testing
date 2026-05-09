#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_DailyBreakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Daily data for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter (long-term trend)
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily high/low for breakout levels (previous day)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Breakout levels: previous day high/low
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Align breakout levels to daily timeframe (no additional delay needed as we use previous day's data)
    breakout_high = align_htf_to_ltf(prices, df_1d, prev_high)
    breakout_low = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume spike filter: volume > 2.0x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(breakout_high[i]) or np.isnan(breakout_low[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above previous day's high with volume spike and above weekly EMA50
            if (price > breakout_high[i] and vol_spike[i] and price > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day's low with volume spike and below weekly EMA50
            elif (price < breakout_low[i] and vol_spike[i] and price < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below previous day's low (mean reversion)
            if price < breakout_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above previous day's high (mean reversion)
            if price > breakout_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals