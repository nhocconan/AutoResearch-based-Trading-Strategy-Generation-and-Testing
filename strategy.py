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
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6h EMA for trend
    ema_fast = pd.Series(close_6h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close_6h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6h ATR for volatility filter
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        rsi_val = rsi_1d_aligned[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when EMA trend turns bearish OR RSI overbought
            if (ema_fast[i] < ema_slow[i]) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when EMA trend turns bullish OR RSI oversold
            if (ema_fast[i] > ema_slow[i]) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: EMA bullish crossover AND RSI not overbought
            if (ema_fast[i] > ema_slow[i]) and (ema_fast[i-1] <= ema_slow[i-1]) and (rsi_val < 70):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: EMA bearish crossover AND RSI not oversold
            elif (ema_fast[i] < ema_slow[i]) and (ema_fast[i-1] >= ema_slow[i-1]) and (rsi_val > 30):
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

name = "6h_EMA_RSI_Trend_Filter"
timeframe = "6h"
leverage = 1.0