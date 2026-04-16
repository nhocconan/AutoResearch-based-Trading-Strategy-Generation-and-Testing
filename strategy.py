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
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR on 12h for stoploss and position sizing
    tr_12h = np.maximum(high_12h - low_12h,
                       np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                  np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === Daily data (HTF) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50 EMA on daily for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Weekly data (HTF) for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 200 EMA on weekly for regime filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_12h[i]
        ema_50_val = ema_50_1d_aligned[i]
        ema_200_val = ema_200_1w_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12h EMA(50) OR weekly regime turns bearish
            if price < ema_50_val or price < ema_200_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12h EMA(50) OR weekly regime turns bullish
            if price > ema_50_val or price > ema_200_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above both 12h EMA(50) and weekly EMA(200) AND daily trend up
            if price > ema_50_val and price > ema_200_val and ema_50_val > ema_200_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # SHORT: Price below both 12h EMA(50) and weekly EMA(200) AND daily trend down
            elif price < ema_50_val and price < ema_200_val and ema_50_val < ema_200_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_EMA50_EMA200_TrendFilter"
timeframe = "12h"
leverage = 1.0