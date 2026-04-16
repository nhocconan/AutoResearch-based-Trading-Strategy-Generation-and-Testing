#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ATR-based stop.
# Long when price breaks above 20-period 1d Donchian high with volume > 1.5x 20-period average.
# Short when price breaks below 20-period 1d Donchian low with volume > 1.5x 20-period average.
# Exit when price closes below Donchian midpoint (long) or above midpoint (short) or ATR stop hit.
# Uses discrete position size 0.25. Donchian provides clear structure, volume confirms breakout legitimacy,
# ATR stop manages risk. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Volume moving average (20-period) on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Donchian and volume MA to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get ATR for stoploss (using 4h data)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
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
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for ATR stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        dm = donchian_mid_aligned[i]
        vol_ma = vol_ma_aligned[i]
        atr_val = atr[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price closes below Donchian midpoint
            if price < dm:
                exit_signal = True
            # ATR stoploss: 2 * ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price closes above Donchian midpoint
            if price > dm:
                exit_signal = True
            # ATR stoploss: 2 * ATR above entry
            elif price > entry_price + 2.0 * atr_val:
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
            
            # LONG: price breaks above Donchian high with volume confirmation
            if price > dh and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian low with volume confirmation
            elif price < dl and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dDonchian20_VolumeConfirmation_ATRStop_V1"
timeframe = "4h"
leverage = 1.0