#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volatility regime filter and volume confirmation
# - Uses 4h Donchian channels (20-period high/low) for breakout signals
# - Filters trades using 1d ATR ratio (current ATR / 50-period ATR) to avoid high volatility periods
# - Requires volume > 1.5x 20-period average for confirmation
# - Exits on opposite Donchian band touch or ATR-based stop (2x ATR)
# - Designed for 15-25 trades/year per symbol to minimize fee drag
# - Works in both bull (breakouts) and bear (mean reversion via volatility filter)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d  # Current ATR relative to 50-day average
    atr_ratio_1d_4h = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Calculate Donchian channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_ratio_1d_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        
        # Volatility filter: only trade when volatility is normal or low (avoid high volatility periods)
        vol_filter = atr_ratio_1d_4h[i] < 1.5
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume surge + vol filter
            if price > donch_high[i] and price > donch_high[i-1] and vol > 1.5 * vol_ma[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + volume surge + vol filter
            elif price < donch_low[i] and price < donch_low[i-1] and vol > 1.5 * vol_ma[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price touches Donchian low OR ATR stop hit (2*ATR from entry)
            if price <= donch_low[i] or price < entry_price - 2.0 * (atr_1d[i] * atr_ratio_1d_4h[i] / 14):  # Approximate 4h ATR
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian high OR ATR stop hit (2*ATR from entry)
            if price >= donch_high[i] or price > entry_price + 2.0 * (atr_1d[i] * atr_ratio_1d_4h[i] / 14):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolRegime_VolumeFilter"
timeframe = "4h"
leverage = 1.0