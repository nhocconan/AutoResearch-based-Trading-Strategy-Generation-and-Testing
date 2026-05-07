#!/usr/bin/env python3
name = "4h_12h_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot points from previous 12h bar (complete bar only)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Use previous 12h bar's complete data to calculate current 12h pivot
    prev_high = high_12h[:-1]
    prev_low = low_12h[:-1]
    prev_close = close_12h[:-1]
    
    if len(prev_high) < 1:
        return np.zeros(n)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    hl_range = prev_high - prev_low
    r2 = pivot + hl_range
    s2 = pivot - hl_range
    
    # Create arrays for each 12h bar (align with 12h bars)
    pivot_per_12h = np.full(len(df_12h), np.nan)
    r1_per_12h = np.full(len(df_12h), np.nan)
    s1_per_12h = np.full(len(df_12h), np.nan)
    r2_per_12h = np.full(len(df_12h), np.nan)
    s2_per_12h = np.full(len(df_12h), np.nan)
    
    # Shift by one 12h bar: current 12h bar gets previous 12h bar's levels
    pivot_per_12h[1:] = pivot
    r1_per_12h[1:] = r1
    s1_per_12h[1:] = s1
    r2_per_12h[1:] = r2
    s2_per_12h[1:] = s2
    
    # Align to 4h timeframe (only complete 12h bars available)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_per_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_per_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_per_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_per_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_per_12h)
    
    # 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50)  # Wait for volume MA, ATR, EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R1, above EMA50, volume spike, volatility not extreme
            vol_condition = volume[i] > vol_ma[i] * 1.5
            vol_not_extreme = atr[i] < np.median(atr[max(0, i-50):i+1]) * 3
            
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50[i] and 
                vol_condition and 
                vol_not_extreme):
                signals[i] = 0.25
                position = 1
            # Short: price < S1, below EMA50, volume spike, volatility not extreme
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50[i] and 
                  vol_condition and 
                  vol_not_extreme):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < S1 or below EMA50 or volatility spike
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_50[i] or
                atr[i] > np.median(atr[max(0, i-50):i+1]) * 4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > R1 or above EMA50 or volatility spike
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_50[i] or
                atr[i] > np.median(atr[max(0, i-50):i+1]) * 4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h trend filter, volume confirmation, and volatility filter.
# 12h Camarilla levels (R1/S1) from previous 12h bar identify key support/resistance.
# Breakout above R1 with volume suggests bullish momentum; breakdown below S1 suggests bearish.
# 4h EMA(50) filter ensures we trade in the direction of the intermediate trend.
# Volume confirmation ensures institutional participation.
# Volatility filter avoids whipsaws during extreme volatility spikes.
# Works in bull markets (buy breakouts above R1 in uptrend) and bear markets (sell breakdowns below S1 in downtrend).
# Position size 0.25 balances risk and keeps trade frequency manageable (~20-40 trades/year).