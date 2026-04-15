#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with Volume Spike and 12h Trend Filter
# Williams %R identifies overbought/oversold conditions. We enter on reversals from extreme levels
# (%R < -80 for long, %R > -20 for short) only when confirmed by volume spike and aligned with
# 12h trend (EMA50 direction). Works in ranging markets (mean reversion) and can catch trend
# continuations when aligned with higher timeframe. Target: 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Williams %R and EMA50 to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i])):
            continue
        
        # Volume spike: current volume > 2x median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Long: Williams %R oversold (< -80) + volume spike + price above 12h EMA50 (uptrend)
        if (williams_r_aligned[i] < -80 and
            volume_spike and
            close[i] > ema_50_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: Williams %R overbought (> -20) + volume spike + price below 12h EMA50 (downtrend)
        elif (williams_r_aligned[i] > -20 and
              volume_spike and
              close[i] < ema_50_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R crosses back through -50 (mean reversion) or opposite extreme
        elif position == 1 and williams_r_aligned[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and williams_r_aligned[i] < -50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Volume_Spike_12hEMA50"
timeframe = "6h"
leverage = 1.0