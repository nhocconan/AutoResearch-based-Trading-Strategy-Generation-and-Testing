#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly: Trend filter (EMA20) ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Daily: Donchian breakout with volume confirmation ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        ema_val = ema20_1w_aligned[i]
        donchian_high = high_max[i]
        donchian_low = low_min[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(donchian_high) or np.isnan(donchian_low) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + weekly uptrend + volume confirmation
            if (close_val > donchian_high and      # Breakout above 20-day high
                close_val > ema_val and            # Price above weekly EMA20 (uptrend)
                vol_ratio_val > 1.5):              # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + weekly downtrend + volume confirmation
            elif (close_val < donchian_low and     # Breakdown below 20-day low
                  close_val < ema_val and          # Price below weekly EMA20 (downtrend)
                  vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reversal
            if close_val < donchian_low:           # Breakdown below 20-day low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reversal
            if close_val > donchian_high:          # Breakout above 20-day high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals