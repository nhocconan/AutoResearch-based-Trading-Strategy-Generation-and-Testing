#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h Volume Spike and 1d ADX Trend Filter
# Takes long when price breaks above 4h Camarilla H3 with 4h volume spike and 1d ADX > 25
# Takes short when price breaks below 4h Camarilla L3 with 4h volume spike and 1d ADX > 25
# Exits when price crosses back below/above 4h Camarilla H4/L4 or volume drops
# Uses 4h for signal direction (Camarilla levels + volume + trend), 1h only for entry timing
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Camarilla levels (based on previous day's range)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, L3 = close - 1.1*(high-low)*1.1/4
    # We use the previous 4h bar's high/low/close to calculate current levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    rang = high_4h - low_4h
    H3 = close_4h + 1.1 * rang * 1.1 / 2
    L3 = close_4h - 1.1 * rang * 1.1 / 4
    H4 = close_4h + 1.1 * rang * 1.1
    L4 = close_4h - 1.1 * rang * 1.1
    
    # Calculate 4h EMA(20) for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    H4_aligned = align_htf_to_ltf(prices, df_4h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_4h, L4)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and ADX calculations
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = vol_4h[i // 4] if i // 4 < len(vol_4h) else vol_4h[-1]  # Current 4h bar volume
        
        if position == 0:
            # Long setup: break above H3 with volume spike and strong trend
            if (price > H3_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                adx_aligned[i] > 25 and                         # Strong trend
                price > ema_4h_aligned[i]):                     # Above 4h EMA (uptrend)
                position = 1
                signals[i] = position_size
            # Short setup: break below L3 with volume spike and strong trend
            elif (price < L3_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                  adx_aligned[i] > 25 and                         # Strong trend
                  price < ema_4h_aligned[i]):                     # Below 4h EMA (downtrend)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below H4 or volume drops
            if price < H4_aligned[i] or vol_4h_current < vol_ma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above L4 or volume drops
            if price > L4_aligned[i] or vol_4h_current < vol_ma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Camarilla_Breakout_4hVolume_1dADX"
timeframe = "1h"
leverage = 1.0