#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_VolumeFilter_v3
Hypothesis: 4h Donchian(20) breakout filtered by 1d EMA50 trend and volume filter (volume > 1.5x median).
Uses discrete position sizing (0.0, ±0.25) to limit trades to ~30/year. ATR trailing stop with 2.0x ATR.
Designed to work in bull/bear via 1d trend alignment and volume filter to avoid false breakouts.
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
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h OHLC for Donchian calculation ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels using previous completed 4h bar
    # Upper = max(high_4h[-20:]), Lower = min(low_4h[-20:])
    high_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (use previous completed 4h bar)
    upper_aligned = align_htf_to_ltf(prices, df_4h, high_max)
    lower_aligned = align_htf_to_ltf(prices, df_4h, low_min)
    
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
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 1.5x 50-period median (more stable than mean)
            volume = prices['volume'].values
            vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
            vol_filter = volume[i] > 1.5 * vol_median[i] if not np.isnan(vol_median[i]) else False
            
            # Long conditions: price > 4h Upper, 1d uptrend, volume filter
            long_breakout = price > upper_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < 4h Lower, 1d downtrend, volume filter
            short_breakout = price < lower_aligned[i]
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
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 4h Lower (support broken)
            elif price < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 4h Upper (resistance broken)
            elif price > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_VolumeFilter_v3"
timeframe = "4h"
leverage = 1.0