#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w volume confirmation and 1d ATR stoploss
# Long when price breaks above 12h Donchian upper band AND 1w volume > 1.5x 20-period average
# Short when price breaks below 12h Donchian lower band AND 1w volume > 1.5x 20-period average
# Uses weekly volume to confirm institutional interest in breakouts
# ATR-based stoploss (2.0x ATR) for risk management
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian channels (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # === 1w Volume Confirmation ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === 1d ATR for stoploss (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or
            np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = upper_12h_aligned[i]
        lower = lower_12h_aligned[i]
        vol_confirm = volume[i] > vol_ma_1w_aligned[i] * 1.5  # 1.5x weekly average volume
        atr_val = atr_1d_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Exit if price drops 2.0*ATR from entry
            if atr_val > 0 and price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if price rises 2.0*ATR from entry
            if atr_val > 0 and price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian upper AND volume confirmation
            if price > upper and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short when: price breaks below Donchian lower AND volume confirmation
            elif price < lower and vol_confirm:
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

name = "12h_Donchian20_1wVolume1.5x_1dATRStop_2.0x"
timeframe = "12h"
leverage = 1.0