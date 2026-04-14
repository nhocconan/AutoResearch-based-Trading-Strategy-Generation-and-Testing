#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR-based stoploss
# Long when price breaks above 4h Donchian channel (20-period) with volume spike
# Short when price breaks below 4h Donchian channel with volume spike
# Exit when price crosses the Donchian midline (10-period average)
# Volume confirmation: current volume > 2.0 * 20-period average volume
# Target: 20-40 trades per year (80-160 over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channel (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[high_4h[0]], high_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[low_4h[0]], low_4h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to lower timeframe (assumes 15m input, but works generically)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    atr_multiplier = 2.0  # ATR stoploss multiplier
    
    # Start after enough data for calculations
    start = 30  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long setup: break above Donchian high with volume spike
            if (price > donchian_high_aligned[i] and 
                vol_current > 2.0 * vol_ma_4h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low with volume spike
            elif (price < donchian_low_aligned[i] and 
                  vol_current > 2.0 * vol_ma_4h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below midline OR stoploss hit
            if (price < donchian_mid_aligned[i] or 
                price < (donchian_high_aligned[i] - atr_multiplier * atr_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above midline OR stoploss hit
            if (price > donchian_mid_aligned[i] or 
                price > (donchian_low_aligned[i] + atr_multiplier * atr_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR_Stop"
timeframe = "4h"
leverage = 1.0