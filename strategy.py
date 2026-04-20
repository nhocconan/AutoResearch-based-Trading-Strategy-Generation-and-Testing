#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h trading based on 12h Donchian channel breakouts with volume confirmation and ATR stop
# - Uses 12h Donchian channel (20-period) as trend filter and breakout levels
# - Entry: price breaks above/below 12h Donchian channel + volume > 1.8x 20-period average
# - Exit: price crosses back through opposite Donchian band or ATR-based stop (2x ATR)
# - Volume confirmation reduces false breakouts, ATR manages risk
# - Target: 25-40 trades per year per symbol (100-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channel (20-period)
    period = 20
    donchian_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    
    # Align Donchian bands to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate ATR for stop loss (using 12h data)
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
        if np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or np.isnan(atr_12h_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above 12h Donchian high + volume surge
            if price > donchian_high_4h[i] and price <= donchian_high_4h[i-1] and vol > 1.8 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below 12h Donchian low + volume surge
            elif price < donchian_low_4h[i] and price >= donchian_low_4h[i-1] and vol > 1.8 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below 12h Donchian low OR ATR stop hit (2*ATR)
            if price < donchian_low_4h[i] or price < entry_price - 2.0 * atr_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h Donchian high OR ATR stop hit (2*ATR)
            if price > donchian_high_4h[i] or price > entry_price + 2.0 * atr_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0