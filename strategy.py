#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - 1w EMA200 as trend filter: long when price above 1w EMA200, short when below
# - Entry: price breaks above/below 1d Donchian channel (20-period) with volume > 1.5x 20-day average
# - Exit: price crosses back through Donchian median or ATR-based stop (2x ATR)
# - Volume confirmation reduces false breakouts
# - Target: 20-30 trades per year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 1d Donchian channel (20-period)
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # Align Donchian levels to 1d index (already aligned since same timeframe)
    # Calculate ATR for stop loss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if NaN in critical values
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(donchian_mid[i]) or np.isnan(atr_1d[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price above 1w EMA200 + breaks above Donchian upper + volume surge
            if price > ema_200_1w_aligned[i] and price > high_max[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price below 1w EMA200 + breaks below Donchian lower + volume surge
            elif price < ema_200_1w_aligned[i] and price < low_min[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid OR ATR stop hit (2*ATR)
            if price < donchian_mid[i] or price < entry_price - 2.0 * atr_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid OR ATR stop hit (2*ATR)
            if price > donchian_mid[i] or price > entry_price + 2.0 * atr_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_1wEMA200_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0