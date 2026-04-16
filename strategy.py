#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Ehlers Fisher Transform with 1d volume confirmation and 12h EMA20 trend filter
# Long when Fisher crosses above -1.5 AND price > 12h EMA20 AND volume > 1.3x 1d average volume
# Short when Fisher crosses below +1.5 AND price < 12h EMA20 AND volume > 1.3x 1d average volume
# ATR trailing stop (2.5x ATR) to manage risk
# Fisher Transform identifies turning points with minimal lag
# EMA20 filter ensures alignment with short-term trend
# Volume confirmation adds conviction to reversals
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee drag

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
    
    # === 1d Fisher Transform ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ehlers Fisher Transform
    # Normalize price to [-1, 1] range over lookback period
    lookback = 10
    hl_range = high_1d - low_1d
    # Avoid division by zero
    hl_range_safe = np.where(hl_range == 0, 1e-10, hl_range)
    value = 2 * ((close_1d - low_1d) / hl_range_safe - 0.5)
    # Clamp to [-0.999, 0.999] to avoid math domain error
    value = np.clip(value, -0.999, 0.999)
    
    # Fisher Transform formula
    fisher = np.zeros_like(value)
    for i in range(1, len(value)):
        fisher[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fisher[i-1]
    
    # Smooth the Fisher transform
    fisher_smooth = pd.Series(fisher).ewm(span=5, adjust=False).values
    
    # Align Fisher Transform to 4h timeframe
    fisher_aligned = align_htf_to_ltf(prices, df_1d, fisher_smooth)
    
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
            np.isnan(fisher_aligned[i]) or
            np.isnan(vol_ma_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_20_aligned[i]
        fisher_val = fisher_aligned[i]
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
            # Fisher crossover signals
            if i > warmup:
                fisher_prev = fisher_aligned[i-1]
                # Long when Fisher crosses above -1.5 AND price > EMA20 AND volume confirmation
                if fisher_prev <= -1.5 and fisher_val > -1.5 and price > ema_val and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when Fisher crosses below +1.5 AND price < EMA20 AND volume confirmation
                elif fisher_prev >= 1.5 and fisher_val < 1.5 and price < ema_val and vol_confirm:
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

name = "4h_FisherTransform_12hEMA20_Volume1.3x_ATRTrail_2.5x"
timeframe = "4h"
leverage = 1.0