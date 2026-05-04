#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# In trending markets (1w price > EMA50): breakout in direction of trend
# In ranging markets (1w price <= EMA50): fade Donchian touches (mean reversion)
# Volume confirmation (>1.3x 20-period EMA) filters low-quality breakouts
# Discrete sizing (0.25) minimizes fee churn. Target: 50-120 trades over 4 years.
# Strategy adapts to bull/bear markets via weekly trend filter and uses 1d primary timeframe.

name = "1d_Donchian20_1wEMA50_Volume_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d Donchian channels (20-period) from previous completed day
    # Shift by 1 to avoid look-ahead (use previous 20 days, not including current)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Upper channel: highest high of previous 20 days
    donchian_high = high_series.shift(1).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of previous 20 days
    donchian_low = low_series.shift(1).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            if close[i] > ema50_1w_aligned[i]:
                # Uptrend: long on break above Donchian high
                if close[i] > donchian_high[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
            else:
                # Downtrend or ranging: short on break below Donchian low
                if close[i] < donchian_low[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel OR trend changes
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if (close[i] <= midpoint or 
                close[i] <= ema50_1w_aligned[i]):  # trend change to down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel OR trend changes
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if (close[i] >= midpoint or 
                close[i] > ema50_1w_aligned[i]):  # trend change to up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals