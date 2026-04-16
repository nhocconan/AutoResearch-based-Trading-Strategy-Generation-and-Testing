#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power < 0 AND EMA13 rising AND volume > 1.5x 12h average
# Short when Bear Power < 0 AND Bull Power < 0 AND EMA13 falling AND volume > 1.5x 12h average
# Uses 12h EMA13 for trend direction and 6h ATR trailing stop
# Focuses on momentum exhaustion and continuation in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h EMA13 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_12h, ema_13)
    
    # === 6h EMA13 for Elder Ray calculation ===
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray Components ===
    bull_power = high - ema_13_6h  # High - EMA13
    bear_power = low - ema_13_6h   # Low - EMA13
    
    # === 12h Volume Confirmation (average over 2 periods of 6h) ===
    vol_ma_12h = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    # === 6h ATR for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position and trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ma_12h[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_13_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_confirm = volume[i] > vol_ma_12h[i] * 1.5
        atr_val = atr[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            if price > highest_since_entry:
                highest_since_entry = price
            if atr_val > 0 and price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            if atr_val > 0 and price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: Bull Power positive, Bear Power negative, EMA13 rising, volume confirmation
            if (bull_val > 0 and bear_val < 0 and 
                ema_val > ema_13_aligned[max(i-1, warmup)] and vol_confirm):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
                continue
            # Short: Bear Power negative, Bull Power negative, EMA13 falling, volume confirmation
            elif (bear_val < 0 and bull_val < 0 and 
                  ema_val < ema_13_aligned[max(i-1, warmup)] and vol_confirm):
                signals[i] = -0.25
                position = -1
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

name = "6h_ElderRay_12hEMA13_Trend_Volume1.5x_ATRTrail_2.5x"
timeframe = "6h"
leverage = 1.0