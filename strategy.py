#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Donchian high(20) with 12h EMA(20) rising and volume > 1.5x average
# - Short when price breaks below Donchian low(20) with 12h EMA(20) falling and volume > 1.5x average
# - Exit when price crosses back through Donchian midpoint or ATR-based stop hit
# - Uses 12h for trend (reduces false signals in chop) and 6h for execution
# - Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(20) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_slope = ema_12h - np.roll(ema_12h, 1)
    ema_12h_slope[0] = 0
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate ATR for stop loss (using 12h data)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 6h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_12h_aligned[i]) or \
           np.isnan(ema_12h_slope_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + 12h EMA rising + volume surge
            if price > donch_high[i] and ema_12h_slope_aligned[i] > 0 and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + 12h EMA falling + volume surge
            elif price < donch_low[i] and ema_12h_slope_aligned[i] < 0 and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint OR ATR stop hit (2*ATR)
            if price < donch_mid[i] or price < entry_price - 2.0 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint OR ATR stop hit (2*ATR)
            if price > donch_mid[i] or price > entry_price + 2.0 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMAFilter_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0