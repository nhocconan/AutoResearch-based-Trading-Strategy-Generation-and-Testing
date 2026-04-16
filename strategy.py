#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper channel AND price > 1w EMA50 AND volume > 2.0x 20-period average volume
# Short when price breaks below 1d Donchian lower channel AND price < 1w EMA50 AND volume > 2.0x 20-period average volume
# ATR trailing stop (2.5x ATR) to manage risk
# Using 1d timeframe for lower trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag
# Volume threshold increased to 2.0x to ensure only high-quality breakouts
# ATR multiplier increased to 2.5 for wider stop in volatile markets
# Designed to work in both bull and bear markets via 1w EMA50 trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w EMA50 (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Donchian Channels (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate Donchian channels
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align HTF data
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # === 1d Volume Spike Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # === 1d ATR for trailing stop (14-period) ===
    # Need 1d close for TR calculation
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_val = upper_channel_aligned[i]
        lower_val = lower_channel_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2.0x average volume for spike
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above upper Donchian channel AND price > 1w EMA50 AND volume spike
            if price > upper_val and price > ema_50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below lower Donchian channel AND price < 1w EMA50 AND volume spike
            elif price < lower_val and price < ema_50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_ATRTrail"
timeframe = "1d"
leverage = 1.0