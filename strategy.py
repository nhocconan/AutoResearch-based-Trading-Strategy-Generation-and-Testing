#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_VolumeFilter_ATRStop_v2
Hypothesis: 4h Donchian(20) breakouts filtered by 1d EMA50 trend and volume spike (>1.5x average).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. ATR-based trailing stop.
Designed for 20-40 trades/year per symbol. Works in bull/bear via 1d trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for Donchian, 1d for trend)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian: upper = max(high,20), lower = min(low,20)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (use previous completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # === 1d EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 1.5x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_filter = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 4h Donchian upper, 1d uptrend, volume filter
            long_breakout = price > donchian_upper_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < 4h Donchian lower, 1d downtrend, volume filter
            short_breakout = price < donchian_lower_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 4h Donchian lower (support broken)
            elif price < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 4h Donchian upper (resistance broken)
            elif price > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_VolumeFilter_ATRStop_v2"
timeframe = "4h"
leverage = 1.0