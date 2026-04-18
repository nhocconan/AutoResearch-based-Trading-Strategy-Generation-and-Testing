#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND 1w EMA(34) rising AND volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) low AND 1w EMA(34) falling AND volume > 1.5x 20-period average
# Exit when price returns to Donchian(20) midpoint or trend weakens.
# Uses price channel breakouts with higher timeframe trend filter and volume confirmation.
# Designed for ~15-25 trades/year per symbol.
name = "12h_Donchian_20_1wEMA34_Volume"
timeframe = "12h"
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
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_rising = ema_1w_34 > np.roll(ema_1w_34, 1)
    ema_1w_34_rising[0] = False
    ema_1w_34_falling = ema_1w_34 < np.roll(ema_1w_34, 1)
    ema_1w_34_falling[0] = False
    ema_1w_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34_rising)
    ema_1w_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34_falling)
    
    # 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_34_rising_aligned[i]) or np.isnan(ema_1w_34_falling_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        don_high = donchian_high[i]
        don_low = donchian_low[i]
        don_mid = donchian_mid[i]
        ema_rising = ema_1w_34_rising_aligned[i]
        ema_falling = ema_1w_34_falling_aligned[i]
        vol_avg = vol_avg_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, 1w EMA rising, volume > 1.5x average
            if price > don_high and ema_rising and vol > 1.5 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, 1w EMA falling, volume > 1.5x average
            elif price < don_low and ema_falling and vol > 1.5 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian mid or 1w EMA stops rising
            if price < don_mid or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian mid or 1w EMA stops falling
            if price > don_mid or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals