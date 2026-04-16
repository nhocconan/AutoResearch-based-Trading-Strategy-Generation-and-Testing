#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1h EMA20 trend filter
# Long when price closes above Donchian upper AND price > 1h EMA20 AND volume > 1.3x 1d average volume
# Short when price closes below Donchian lower AND price < 1h EMA20 AND volume > 1.3x 1d average volume
# ATR trailing stop (2.5x ATR) to manage risk
# Donchian(20) provides clear trend-following structure with proven edge in both bull/bear markets
# EMA20 filter ensures alignment with short-term trend to avoid counter-trend entries
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
    
    # === 1h EMA20 trend filter ===
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_20 = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1h, ema_20)
    
    # === 4h Donchian(20) channels ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Volume Confirmation ===
    vol_ma_1d = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods of 1h = 1d (4h data)
    
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
    warmup = 50
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_20_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.3  # 1.3x average volume for confirmation
        atr_val = atr[i]
        
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
            # Long when: price closes above Donchian upper AND price > EMA20 AND volume confirmation
            if price > upper and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price closes below Donchian lower AND price < EMA20 AND volume confirmation
            elif price < lower and price < ema_val and vol_confirm:
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

name = "4h_Donchian20_1hEMA20_Volume1.3x_ATRTrail_2.5x"
timeframe = "4h"
leverage = 1.0