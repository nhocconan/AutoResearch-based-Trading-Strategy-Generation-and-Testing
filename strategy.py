#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_VolumeFilter_V1
Hypothesis: 6h Donchian(20) breakout filtered by 1w EMA50 trend and 6h volume spike (>1.5x average).
Enter long when price breaks above 6h Donchian upper with 1w uptrend and volume spike.
Enter short when price breaks below 6h Donchian lower with 1w downtrend and volume spike.
Exit on ATR(14) trailing stop (2.0*ATR) or opposite Donchian level break.
Designed for low trade frequency (<30 trades/year) to minimize fee drag.
Works in bull/bear via 1w trend alignment and volume spike filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 6h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian: upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 1w EMA50 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === ATR (14-period) for stoploss ===
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 1.5x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 6h Donchian upper, 1w uptrend, volume spike
            long_breakout = price > donchian_upper[i]
            long_trend = price > ema_50_1w_aligned[i]
            
            # Short conditions: price < 6h Donchian lower, 1w downtrend, volume spike
            short_breakout = price < donchian_lower[i]
            short_trend = price < ema_50_1w_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 6h Donchian lower (support broken)
            elif price < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 6h Donchian upper (resistance broken)
            elif price > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0