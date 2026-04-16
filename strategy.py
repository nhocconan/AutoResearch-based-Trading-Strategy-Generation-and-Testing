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
    
    # === 1d data (HTF for key levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4d data (HTF for trend filter) ===
    df_4d = get_htf_data(prices, '4d')
    close_4d = df_4d['close'].values
    
    # === Calculate 1d price channels (previous day's high/low) ===
    # Using previous day's OHLC for support/resistance
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Support and Resistance levels (previous day's high/low)
    resistance = prev_high_1d
    support = prev_low_1d
    
    # Align to 1h timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance)
    support_aligned = align_htf_to_ltf(prices, df_1d, support)
    
    # === 4d EMA50 for trend filter ===
    ema_50_4d = pd.Series(close_4d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4d_aligned = align_htf_to_ltf(prices, df_4d, ema_50_4d)
    
    # === Volume confirmation (1h) ===
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    # === Session filter (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN or outside session
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(ema_50_4d_aligned[i]) or np.isnan(vol_ratio[i]) or
            not session_mask[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        resistance_val = resistance_aligned[i]
        support_val = support_aligned[i]
        ema_50_4d_val = ema_50_4d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below support (stop) or hits resistance*1.02 (take profit)
            if price < support_val or price > resistance_val * 1.02:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above resistance (stop) or hits support*0.98 (take profit)
            if price > resistance_val or price < support_val * 0.98:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above resistance with volume AND above 4d EMA50 (uptrend)
            if (price > resistance_val) and (price > ema_50_4d_val) and (vol_ratio_val > 1.5):
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price breaks below support with volume AND below 4d EMA50 (downtrend)
            elif (price < support_val) and (price < ema_50_4d_val) and (vol_ratio_val > 1.5):
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Resistance_Support_Breakout_Volume_EMA50_4d_Session"
timeframe = "1h"
leverage = 1.0