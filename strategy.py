#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# - Uses 1d EMA100 as trend filter: price must be above EMA100 for long, below for short
# - Entry: price breaks above Donchian(20) high + volume > 1.5x 20-period average + EMA filter
# - Exit: price breaks below Donchian(20) low OR ATR stop hit (2x ATR)
# - Combines trend following (EMA), breakout (Donchian), volume confirmation, and risk control
# - Target: 20-30 trades per year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA100 on 1d data
    ema_100 = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_4h = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # Calculate ATR for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_100_4h[i]) or np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price above EMA100 + breaks above Donchian high + volume surge
            if price > ema_100_4h[i] and price > donch_high[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price below EMA100 + breaks below Donchian low + volume surge
            elif price < ema_100_4h[i] and price < donch_low[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR ATR stop hit (2*ATR)
            if price < donch_low[i] or price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR ATR stop hit (2*ATR)
            if price > donch_high[i] or price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA100_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0