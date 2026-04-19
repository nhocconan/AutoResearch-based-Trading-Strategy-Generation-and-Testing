#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Long when: price breaks above Donchian(20) upper, price > weekly pivot, volume > 1.5x avg
# Short when: price breaks below Donchian(20) lower, price < weekly pivot, volume > 1.5x avg
# Weekly pivot provides higher timeframe directional bias to avoid counter-trend trades.
# Volume confirms breakout strength. Designed for ~15-25 trades/year per symbol.
name = "6h_Donchian_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot direction (higher timeframe bias)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Donchian channels (20-period) on 6h data
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Get current levels
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        weekly_pivot = pp_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price > Donchian upper with volume and above weekly pivot
            if price > upper and vol > 1.5 * vol_ma and price > weekly_pivot:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian lower with volume and below weekly pivot
            elif price < lower and vol > 1.5 * vol_ma and price < weekly_pivot:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian lower or weekly pivot
            if price <= lower or price <= weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian upper or weekly pivot
            if price >= upper or price >= weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals