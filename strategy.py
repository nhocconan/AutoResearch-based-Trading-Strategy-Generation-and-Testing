#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume spike filter and 1d trend filter
# Long when price breaks above Donchian upper channel AND 12h volume > 2.0x 20-period average AND price > 1d EMA50
# Short when price breaks below Donchian lower channel AND 12h volume > 2.0x 20-period average AND price < 1d EMA50
# ATR trailing stop (2.5x ATR) to manage risk
# Donchian breakout captures momentum, volume spike confirms conviction, 1d EMA50 filters for higher-timeframe trend
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate Donchian Channel
    upper_donch = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF data
    upper_donch_aligned = align_htf_to_ltf(prices, df_12h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_12h, lower_donch)
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 6h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_donch_aligned[i]) or
            np.isnan(lower_donch_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_donch_val = upper_donch_aligned[i]
        lower_donch_val = lower_donch_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2.0x average volume
        atr_val = atr_6h[i]
        
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
        
        # === EXIT LOGIC (Donchian middle touch) ===
        if position == 1:  # Long position
            # Exit when price touches or crosses below Donchian middle
            donch_middle = (upper_donch_val + lower_donch_val) / 2.0
            if price <= donch_middle:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches or crosses above Donchian middle
            donch_middle = (upper_donch_val + lower_donch_val) / 2.0
            if price >= donch_middle:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above upper Donchian AND volume confirmation AND price > 1d EMA50
            if price > upper_donch_val and vol_confirm and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below lower Donchian AND volume confirmation AND price < 1d EMA50
            elif price < lower_donch_val and vol_confirm and price < ema_50_val:
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

name = "6h_Donchian20_12hVolSpike_1dEMA50_ATRTrail"
timeframe = "6h"
leverage = 1.0