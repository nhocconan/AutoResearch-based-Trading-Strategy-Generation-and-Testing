#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA34 trend filter + volume confirmation
# Williams Alligator identifies trend absence/presence via smoothed medians (Jaw/Teeth/Lips).
# In chop (Alligator sleeping): fade extremes. In trend (Alligator awakening): follow breakouts.
# 1w EMA34 ensures alignment with weekly trend. Volume confirms participation.
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 trades over 4 years.
# Works in bull markets (follow weekly trend breakouts) and bear markets (fade daily extremes in chop).

name = "1d_WilliamsAlligator_1wEMA34_Volume_v1"
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
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw (Blue): 13-period SMMA shifted 8 bars
    # Teeth (Red): 8-period SMMA shifted 5 bars
    # Lips (Green): 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Using high for Jaw (typical Alligator uses median price)
    teeth = smma(high, 8)  # Using high for Teeth
    lips = smma(high, 5)   # Using high for Lips
    
    # Shift to avoid look-ahead (Alligator uses future data in calculation)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First shifted values become NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Alligator state: 
    # Sleeping (chop): Jaw > Teeth > Lips OR Lips > Teeth > Jaw (all intertwined)
    # Awakening (trend): Lips > Teeth > Jaw (bullish) OR Jaw > Teeth > Lips (bearish)
    # We'll use the cross of Lips and Jaw as trigger
    lips_above_jaw = lips > jaw
    lips_below_jaw = lips < jaw
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(lips[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips[i]
        curr_jaw = jaw[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike
            if curr_volume_spike:
                # Bullish: Lips crosses above Jaw AND price above weekly EMA34
                if curr_lips > curr_jaw and lips_above_jaw[i] and not lips_above_jaw[i-1] and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips crosses below Jaw AND price below weekly EMA34
                elif curr_lips < curr_jaw and lips_below_jaw[i] and not lips_below_jaw[i-1] and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Lips crosses below Jaw OR loses weekly trend
            if curr_lips < curr_jaw or curr_close < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Lips crosses above Jaw OR loses weekly trend
            if curr_lips > curr_jaw or curr_close > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals