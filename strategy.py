#!/usr/bin/env python3
"""
6h_12h_1d_Supertrend_Squeeze_MeanReversion_v1
Hypothesis: Combine Supertrend trend filter with Bollinger squeeze on 12h to identify mean-reversion opportunities in ranging markets, using 1d for regime filtering. Only trade when volatility is low (squeeze) and price is at Bollinger band extremes, with Supertrend confirming absence of strong trend. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Supertrend_Squeeze_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Bollinger Bands (20, 2.0) on 12h
    sma_20 = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i < 19:
            sma_20[i] = np.nan
        else:
            sma_20[i] = np.mean(close_12h[i-19:i+1])
    
    std_20 = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i < 19:
            std_20[i] = np.nan
        else:
            std_20[i] = np.std(close_12h[i-19:i+1])
    
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # normalized width
    
    # Bollinger Squeeze: BB width below 20-period mean
    bb_width_ma = np.zeros_like(bb_width)
    for i in range(len(bb_width)):
        if i < 19:
            bb_width_ma[i] = np.nan
        else:
            bb_width_ma[i] = np.mean(bb_width[i-19:i+1])
    
    squeeze = bb_width < bb_width_ma  # true when in low volatility
    
    # === 1D DATA FOR REGIME ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) for trend detection on 1d
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Wilder smoothing
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            result[period-1] = np.nanmean(arr[1:period])
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = smooth_wilder(tr, period)
        dm_plus_smooth = smooth_wilder(dm_plus, period)
        dm_minus_smooth = smooth_wilder(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
        di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # === 12H SUPERTREND (10, 3.0) ===
    def supertrend(high, low, close, atr_period=10, multiplier=3.0):
        if len(high) < atr_period:
            return np.full(len(high), np.nan), np.full(len(high), np.nan)
        
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.zeros_like(tr)
        atr[atr_period-1] = np.nanmean(tr[1:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
        
        # Upper and Lower Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        # Supertrend
        supertrend = np.zeros_like(close)
        direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
        
        supertrend[0] = 0
        direction[0] = 1
        
        for i in range(1, len(close)):
            if close[i] > upper_band[i-1]:
                direction[i] = 1
            elif close[i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
                if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                    lower_band[i] = lower_band[i-1]
                if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                    upper_band[i] = upper_band[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        return supertrend, direction
    
    st, st_dir = supertrend(high_12h, low_12h, close_12h, 10, 3.0)
    
    # Align 12h indicators to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze.astype(float))
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_12h, sma_20)
    st_aligned = align_htf_to_ltf(prices, df_12h, st)
    st_dir_aligned = align_htf_to_ltf(prices, df_12h, st_dir)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20-period for 6h = ~5 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(sma_20_aligned[i]) or 
            np.isnan(st_aligned[i]) or np.isnan(st_dir_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Conditions for mean reversion setup
        # 1. Bollinger squeeze (low volatility environment)
        vol_squeeze = squeeze_aligned[i] > 0.5  # boolean as float
        
        # 2. No strong trend (ADX < 25 on 1d)
        no_strong_trend = adx_1d_aligned[i] < 25
        
        # 3. Supertrend flat or changing direction (no strong directional bias)
        # Allow small counter-trend moves against supertrend for mean reversion
        supertrend_neutral = True  # we'll use price relative to bands instead
        
        # Mean reversion entries at Bollinger Band extremes
        long_setup = (close[i] <= lower_bb_aligned[i]) and vol_squeeze and no_strong_trend
        short_setup = (close[i] >= upper_bb_aligned[i]) and vol_squeeze and no_strong_trend
        
        # Exit when price returns to middle (SMA20) or opposite band
        exit_long = close[i] >= sma_20_aligned[i]
        exit_short = close[i] <= sma_20_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals