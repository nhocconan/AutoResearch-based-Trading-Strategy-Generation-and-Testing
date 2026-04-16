#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR trailing stop.
# Long when price breaks above 1d Donchian upper channel, volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian lower channel, volume > 1.5x 20-period average.
# Exit when price reverses to midpoint of Donchian channel or ATR stoploss hit.
# Uses discrete position size 0.25. Donchian provides clear structure, volume confirms breakout strength,
# ATR stop manages risk. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian Channel (20) ===
    donchian_len = 20
    dc_upper = pd.Series(high_1d).rolling(window=donchian_len, min_periods=donchian_len).max().values
    dc_lower = pd.Series(low_1d).rolling(window=donchian_len, min_periods=donchian_len).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Align 1d Donchian channels to 12h timeframe
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower)
    dc_middle_aligned = align_htf_to_ltf(prices, df_1d, dc_middle)
    
    # === 1d Indicators: ATR (14) for stoploss ===
    atr_len = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=atr_len, min_periods=atr_len).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d Indicators: Volume moving average (20-period) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or 
            np.isnan(dc_middle_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        dc_upper_val = dc_upper_aligned[i]
        dc_lower_val = dc_lower_aligned[i]
        dc_middle_val = dc_middle_aligned[i]
        atr_val = atr_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price drops below Donchian middle or ATR stoploss hit
            if price < dc_middle_val or price < stop_price:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian middle or ATR stoploss hit
            if price > dc_middle_val or price > stop_price:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # LONG: price breaks above Donchian upper with volume confirmation
            if price > dc_upper_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
                stop_price = price - 2.0 * atr_val  # 2x ATR stoploss
            
            # SHORT: price breaks below Donchian lower with volume confirmation
            elif price < dc_lower_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
                stop_price = price + 2.0 * atr_val  # 2x ATR stoploss
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dDonchian20_VolumeConfirmation_ATRStop_V1"
timeframe = "12h"
leverage = 1.0