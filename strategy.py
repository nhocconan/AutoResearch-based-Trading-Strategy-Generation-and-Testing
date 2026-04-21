#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_WeeklyPivotDirection_VolumeSpike_ATRStop_v1
Hypothesis: 6h Donchian(20) breakouts filtered by 1d EMA50 trend and weekly pivot direction (from 1w HTF).
Only take longs when price > weekly pivot (bullish bias) and shorts when price < weekly pivot (bearish bias).
Volume confirmation (>2x 20-period average) avoids false breakouts. ATR-based trailing stop with 2.0x ATR.
Designed for 12-37 trades/year per symbol (~50-150 total over 4 years) to minimize fee drag.
Works in bull/bear via 1d trend alignment and weekly pivot directional filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Donchian calculation and EMA, 1w for pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d OHLC for Donchian(20) calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels using previous completed 1d bar
    # Upper = max(high_1d over last 20 periods)
    # Lower = min(low_1d over last 20 periods)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (use previous completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1w pivot point (standard calculation) for directional bias ===
    # Pivot = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align to 1w timeframe (use previous completed 1w bar)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 2x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 1d Donchian upper, 1d uptrend, price > weekly pivot (bullish bias), volume spike
            long_breakout = price > donchian_upper_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            long_bias = price > pivot_1w_aligned[i]
            
            # Short conditions: price < 1d Donchian lower, 1d downtrend, price < weekly pivot (bearish bias), volume spike
            short_breakout = price < donchian_lower_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            short_bias = price < pivot_1w_aligned[i]
            
            # Entry logic - ONLY enter on volume spike + trend alignment + pivot bias
            if long_breakout and long_trend and long_bias and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and short_bias and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 1d Donchian lower (support broken)
            elif price < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 1d Donchian upper (resistance broken)
            elif price > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_WeeklyPivotDirection_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0