#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend filter, 1d ATR for volatility filter,
# and 4h Donchian channel breakout with volume confirmation.
# Supertrend identifies trend direction with ATR-based stop levels.
# Donchian breakout captures momentum in trending markets.
# Volume confirmation ensures breakout strength.
# ATR filter avoids low-volatility choppy periods.
# Works in bull/bear: Supertrend adapts to trends, Donchian captures breakouts.
# Target: 20-50 total trades over 4 years (5-12/year) with 0.25 position sizing.

name = "4h_Supertrend_Donchian_Volume_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Supertrend ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upperband[i-1]:
            direction[i] = 1
        elif close_12h[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # Align Supertrend to 4h
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate 1d ATR for volatility filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channel (20-period) ONCE before loop
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_4h
    donchian_low = low_4h
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility (ATR below 50% of its 50-period MA)
        atr_ma_50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
        volatility_filter = atr_1d_aligned[i] > (0.5 * atr_ma_50[i]) if not np.isnan(atr_ma_50[i]) else False
        
        if position == 0:
            # Long: price breaks above Donchian high, uptrend, volume confirmation, sufficient volatility
            if (close[i] > donchian_high[i] and 
                direction_aligned[i] == 1 and 
                volume_filter[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, downtrend, volume confirmation, sufficient volatility
            elif (close[i] < donchian_low[i] and 
                  direction_aligned[i] == -1 and 
                  volume_filter[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend changes to downtrend
            if close[i] < donchian_low[i] or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend changes to uptrend
            if close[i] > donchian_high[i] or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals