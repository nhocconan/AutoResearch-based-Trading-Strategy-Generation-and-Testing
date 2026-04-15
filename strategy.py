#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation
# Designed for low trade frequency (target 15-30/year) with clear trend-following logic
# Works in both bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend) markets
# Uses Donchian channels from 12h, volume spike to confirm breakout strength, and weekly EMA for trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) from 12h data
    # Using previous period's data to avoid look-ahead
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[np.nan], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[np.nan], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on 12h)
    vol_avg = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to main timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + uptrend + volume spike
        if (high[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + downtrend + volume spike
        elif (low[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price retracement to midpoint
        elif position == 1 and close[i] < (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_1wEMA_Volume_Trend"
timeframe = "12h"
leverage = 1.0