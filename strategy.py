#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout + weekly EMA50 trend filter + volume confirmation
# Long when price breaks above 20-day high AND price > weekly EMA50 AND volume > 1.3x 10-day average volume
# Short when price breaks below 20-day low AND price < weekly EMA50 AND volume > 1.3x 10-day average volume
# ATR trailing stop (2.5x ATR) for risk management
# Donchian channels capture breakouts, weekly EMA filters trend direction, volume confirms conviction
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Donchian(20) channels ===
    # 20-day high
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # 20-day low
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume confirmation: 10-day average volume ===
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # === Daily ATR(14) for trailing stop ===
    high_d = high
    low_d = low
    close_d = close
    
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position and trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_10[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma_10[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.3x 10-day average
        vol_confirm = volume[i] > vol_ma_val * 1.3
        
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
            # Long when: price breaks above 20-day high AND price > weekly EMA50 AND volume confirmation
            if price > high_20_val and price > ema_50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below 20-day low AND price < weekly EMA50 AND volume confirmation
            elif price < low_20_val and price < ema_50_val and vol_confirm:
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

name = "1d_Donchian20_1wEMA50_Volume1.3x_ATRTrail_2.5x"
timeframe = "1d"
leverage = 1.0