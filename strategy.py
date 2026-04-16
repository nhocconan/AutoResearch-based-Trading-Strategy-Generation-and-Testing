#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ATR ratio > 0.8 (sufficient volatility) AND 4h volume > 1.3x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d ATR ratio > 0.8 AND 4h volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, ATR filter avoids low-volatility whipsaws, volume confirms participation.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 1d data once before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough for ATR calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ATR (14-period) and ATR Ratio (current / 20-period MA) ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma
    
    # Align 1d ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 20 for volume MA, 34 for ATR ratio)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_spike = volume_spike[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian low or volatility drops
            if price < lower_channel or atr_ratio_val < 0.8:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian high or volatility drops
            if price > upper_channel or atr_ratio_val < 0.8:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND sufficient volatility AND volume spike
            if price > upper_channel and atr_ratio_val > 0.8 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND sufficient volatility AND volume spike
            elif price < lower_channel and atr_ratio_val > 0.8 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0