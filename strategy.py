#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above upper Donchian band, short when breaks below lower band
# Requires: ATR(14) > 0.5 * ATR(50) to filter low volatility, volume > 1.2x 20-period average
# Exit: opposite Donchian breach or volatility collapse (ATR < 0.3 * ATR(50))
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for ATR filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h Donchian(20) channels ===
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # === 1d ATR filter ===
    tr1 = pd.Series(high_1d).diff()
    tr2 = abs(pd.Series(high_1d).diff())
    tr3 = abs(pd.Series(low_1d).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr.rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_50 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === 12h volume confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_band = highest_high_20[i]
        lower_band = lowest_low_20[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ratio = vol_ratio_12h[i]
        
        # === STOPLOSS / EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below lower Donchian band or volatility collapses
            if price < lower_band or atr_ratio_val < 0.3:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian band or volatility collapses
            if price > upper_band or atr_ratio_val < 0.3:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require sufficient volatility and volume confirmation
            if atr_ratio_val > 0.5 and vol_ratio > 1.2:
                # Buy when price breaks above upper Donchian band
                if price > upper_band:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Sell when price breaks below lower Donchian band
                elif price < lower_band:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_ATR_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0