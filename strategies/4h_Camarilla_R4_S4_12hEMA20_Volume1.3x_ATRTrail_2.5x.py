# 4h_Camarilla_R4_S4_12hEMA20_Volume1.3x_ATRTrail_2.5x
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d volume confirmation and 12h EMA20 trend filter
# Long when price closes above Camarilla R4 AND price > 12h EMA20 AND volume > 1.3x 1d average volume
# Short when price closes below Camarilla S4 AND price < 12h EMA20 AND volume > 1.3x 1d average volume
# ATR trailing stop (2.5x ATR) to manage risk
# Camarilla R4/S4 levels from daily pivot provide strong institutional breakout levels
# EMA20 filter ensures alignment with short-term trend
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
    
    # === 12h EMA20 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # === 1d Camarilla levels (from previous day) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points from previous day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (focus on R4/S4 for strong breakouts)
    r4 = close_1d + (range_1d * 1.1 / 2.0)
    s4 = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_20_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
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
            # Long when: price closes above Camarilla R4 AND price > EMA20 AND volume confirmation
            if price > r4_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price closes below Camarilla S4 AND price < EMA20 AND volume confirmation
            elif price < s4_val and price < ema_val and vol_confirm:
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

name = "4h_Camarilla_R4_S4_12hEMA20_Volume1.3x_ATRTrail_2.5x"
timeframe = "4h"
leverage = 1.0