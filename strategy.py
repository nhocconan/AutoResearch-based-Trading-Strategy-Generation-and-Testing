# 12h_Camarilla_R1_S1_1dEMA34_Volume1.5x_ATRTrail_2.0x
# Strategy: Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above S1 (support 1) and closes above EMA34 with volume > 1.5x average
# Short when price breaks below R1 (resistance 1) and closes below EMA34 with volume > 1.5x average
# ATR trailing stop (2.0x) for risk management
# Designed for 12h timeframe to capture multi-day trends while minimizing trade frequency
# Works in bull markets via breakouts and in bear markets via mean reversion at pivot levels

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === 1d Volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # === 1d High/Low for Camarilla pivot calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # Using same close for pivot calculation
    
    # Calculate Camarilla levels (based on previous day's range)
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    range_1d = high_1d - low_1d
    r1 = close_1d_prev + range_1d * 1.1 / 12
    s1 = close_1d_prev - range_1d * 1.1 / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h ATR for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and exit levels
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        vol_avg_val = vol_avg_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        vol_confirm = volume[i] > vol_avg_val * 1.5
        
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
            # Long when: price closes above S1 AND above EMA34 AND volume confirmation
            if close[i] > s1_val and close[i] > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price closes below R1 AND below EMA34 AND volume confirmation
            elif close[i] < r1_val and close[i] < ema_val and vol_confirm:
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

name = "12h_Camarilla_R1_S1_1dEMA34_Volume1.5x_ATRTrail_2.0x"
timeframe = "12h"
leverage = 1.0