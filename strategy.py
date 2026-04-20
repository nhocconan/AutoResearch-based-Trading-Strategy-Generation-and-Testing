# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout above 12h EMA50 with volume confirmation and ATR stop
# - Uses 12h EMA50 as trend filter: price must be above EMA50 for long, below for short
# - Entry: price breaks above 12h EMA50 + volume > 1.5x 20-period average
# - Exit: price crosses back below 12h EMA50 or ATR-based stop hit (2x ATR)
# - Volume confirmation reduces false breakouts
# - ATR stop manages risk during adverse moves
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate ATR for stop loss (using 12h data)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_12h_4h = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_50_4h[i]) or np.isnan(atr_12h_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price above EMA50 + breaks above EMA50 + volume surge
            if price > ema_50_4h[i] and price > ema_50_4h[i-1] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price below EMA50 + breaks below EMA50 + volume surge
            elif price < ema_50_4h[i] and price < ema_50_4h[i-1] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below EMA50 OR ATR stop hit (2*ATR)
            if price < ema_50_4h[i] or price < entry_price - 2.0 * atr_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA50 OR ATR stop hit (2*ATR)
            if price > ema_50_4h[i] or price > entry_price + 2.0 * atr_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA50_Breakout_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0