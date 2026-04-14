#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Weekly pivot (from Monday weekly close) determines long/short bias
# Donchian breakout in direction of weekly bias with volume > 1.5x 20-period average
# Works in bull/bear markets: weekly pivot adapts to trend, volume filters false breakouts
# Uses 60% position size to balance risk/reward

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot calculation
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # Bias: long if close > pivot, short if close < pivot
    typical_price = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    pivot = typical_price.values
    
    # Align pivot to 6h timeframe (use prior week's pivot)
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    
    # Load daily data ONCE for volume average
    df_d = get_htf_data(prices, '1d')
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(df_d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_d, vol_ma_20)
    
    # Calculate 6h Donchian channels (20 periods)
    donch_len = 20
    upper = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lower = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.30  # 30% position size
    
    # Start after enough data for calculations
    start = max(50, donch_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(upper[i]) or 
            np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly bias
        weekly_bias_long = close[i] > pivot_aligned[i]
        weekly_bias_short = close[i] < pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Enter long: weekly bias long + Donchian breakout up + volume confirmation
            if weekly_bias_long and close[i] > upper[i-1] and vol_confirm:
                position = 1
                signals[i] = position_size
            # Enter short: weekly bias short + Donchian breakdown down + volume confirmation
            elif weekly_bias_short and close[i] < lower[i-1] and vol_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price retouches Donchian lower OR weekly bias flips
            if close[i] < lower[i] or not weekly_bias_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price retouches Donchian upper OR weekly bias flips
            if close[i] > upper[i] or not weekly_bias_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_Volume_v1"
timeframe = "6h"
leverage = 1.0