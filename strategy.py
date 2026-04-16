#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) with 1d volume spike and ATR-based stoploss.
# Long when price breaks above Donchian upper (20) AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower (20) AND 1d volume > 1.5x 20-period average.
# Exit when price crosses Donchian midpoint (10-period) OR ATR stoploss triggered.
# Uses discrete position size 0.25. Donchian provides structure, volume confirms breakout strength,
# ATR stoploss manages risk. Designed to capture sustained moves in both bull and bear markets.
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
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA calculation
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian/volume MA, 14 for ATR)
    warmup = 30
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        atr_val = atr[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses Donchian midpoint OR ATR stoploss triggered (2*ATR below entry)
            if price < mid or price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses Donchian midpoint OR ATR stoploss triggered (2*ATR above entry)
            if price > mid or price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND 1d volume spike
            if price > upper and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND 1d volume spike
            elif price < lower and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            # Hold current position
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0