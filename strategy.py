#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation.
# Long when: price breaks above Donchian upper (20) and price > 1w EMA(20) and volume > 1.5x 20-day avg volume
# Short when: price breaks below Donchian lower (20) and price < 1w EMA(20) and volume > 1.5x 20-day avg volume
# Exit when price crosses back through Donchian midpoint or volume drops below average.
# Uses price breakout for entry, weekly trend for direction filter, and volume surge for confirmation.
# Designed for ~10-20 trades/year per symbol to minimize fee drag.
name = "1d_Donchian_20_1wEMA20_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_20_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_20)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_20_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1w = ema_1w_20_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        vol_surge = volume_surge[i]
        
        if position == 0:
            # Long: breakout above upper, price above 1w EMA, volume surge
            if price > upper and price > ema_1w and vol_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower, price below 1w EMA, volume surge
            elif price < lower and price < ema_1w and vol_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint or volume drops below average
            if price < mid or volume[i] < vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint or volume drops below average
            if price > mid or volume[i] < vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals