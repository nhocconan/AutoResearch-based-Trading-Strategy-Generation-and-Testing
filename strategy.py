#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian high AND price > 12h EMA50 AND volume > 1.5x 12h average volume
# Short when price breaks below Donchian low AND price < 12h EMA50 AND volume > 1.5x 12h average volume
# ATR trailing stop (2.0x ATR) to manage risk
# Donchian channels provide clear breakout levels with strong trend-following properties
# EMA50 filter ensures alignment with intermediate-term trend
# Volume confirmation adds conviction to breakouts
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h EMA50 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === 12h Average Volume for confirmation ===
    vol_ma_12h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods of 2h = 12h (4h data)
    
    # === 4h Donchian Channels (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma_12h[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        vol_confirm = volume[i] > vol_ma_12h[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND price > EMA50 AND volume confirmation
            if price > upper and price > ema_val and vol_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Donchian low AND price < EMA50 AND volume confirmation
            elif price < lower and price < ema_val and vol_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume1.5x_ATRTrail_2.0x"
timeframe = "4h"
leverage = 1.0