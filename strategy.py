#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Calculate 1d ATR(14) ===
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Calculate 1d EMA(50) for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Calculate 4h Donchian Channel (20) ===
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === Calculate 4h RSI(14) for momentum filter ===
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(rsi_14[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_4h[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        highest_val = highest_20[i]
        lowest_val = lowest_20[i]
        rsi_val = rsi_14[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 4h Donchian lower band OR RSI > 70 (overbought)
            if (price < lowest_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h Donchian upper band OR RSI < 30 (oversold)
            if (price > highest_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 4h Donchian upper band AND 1d EMA(50) up AND RSI between 40 and 60
            if (price > highest_val) and (ema_50_val > ema_50_1d_aligned[i-1]) and (40 <= rsi_val <= 60):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 4h Donchian lower band AND 1d EMA(50) down AND RSI between 40 and 60
            elif (price < lowest_val) and (ema_50_val < ema_50_1d_aligned[i-1]) and (40 <= rsi_val <= 60):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_1dEMA50_RSI_Filter"
timeframe = "4h"
leverage = 1.0