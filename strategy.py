#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above 12h Donchian upper channel (20-period) with volume > 1.5x 20-period average.
# Short when price breaks below 12h Donchian lower channel with volume > 1.5x 20-period average.
# Exit when price touches the opposite Donchian channel or ATR stoploss is hit.
# Uses discrete position size 0.25. Donchian provides clear structure, volume confirms breakout validity,
# ATR stoploss manages risk. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    donchian_window = 20
    upper_channel = pd.Series(high_12h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low_12h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Align 12h Donchian channels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Volume moving average (20-period) on 12h
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # ATR for stoploss calculation (14-period on 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        vol_ma = vol_ma_aligned[i]
        atr = atr_14[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price touches lower Donchian channel or ATR stoploss hit
            if price <= lower or price <= entry_price - 2.5 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price touches upper Donchian channel or ATR stoploss hit
            if price >= upper or price >= entry_price + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: price breaks above upper Donchian channel with volume confirmation
            if price > upper and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below lower Donchian channel with volume confirmation
            elif price < lower and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_12hDonchian20_VolumeConfirmation_ATRStop_V1"
timeframe = "4h"
leverage = 1.0