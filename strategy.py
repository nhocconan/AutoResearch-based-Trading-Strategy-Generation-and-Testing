#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Uses 1w EMA200 as long-term trend filter: price must be above for long, below for short
# - Entry: price breaks above/below 1d Donchian channel (20-period high/low) + volume > 1.5x 20-day average
# - Exit: price crosses back to the middle of the Donchian channel or ATR-based stop (1.5x ATR)
# - Volume confirmation reduces false breakouts in low-liquidity periods
# - ATR stop manages risk during adverse moves
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Load 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Volume confirmation: 20-day average on 1d data
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to lower timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_1d = align_htf_to_ltf(prices, df_1d, donchian_mid)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d price and volume data (aligned to lower timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(donchian_mid_1d[i]) or np.isnan(ema_200_1d[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 1w EMA200 + volume surge
            if price > donchian_high_1d[i] and price > ema_200_1d[i] and vol > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + below 1w EMA200 + volume surge
            elif price < donchian_low_1d[i] and price < ema_200_1d[i] and vol > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid OR ATR stop hit (1.5*ATR)
            if price < donchian_mid_1d[i] or price < entry_price - 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid OR ATR stop hit (1.5*ATR)
            if price > donchian_mid_1d[i] or price > entry_price + 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA200_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0