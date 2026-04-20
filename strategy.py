#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WeeklyTrend_DailyBreakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w: Weekly trend filter using EMA50 ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily: Price breakout and volume confirmation ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high/low of prior 20 days for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_trend = ema50_1w_aligned[i]
        high_break = high_20[i-1]  # prior 20-day high
        low_break = low_20[i-1]    # prior 20-day low
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(high_break) or np.isnan(low_break) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high in weekly uptrend with volume
            if (high_val > high_break and 
                close_val > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low in weekly downtrend with volume
            elif (low_val < low_break and 
                  close_val < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low or trend changes
            if low_val < low_break or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high or trend changes
            if high_val > high_break or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals