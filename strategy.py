#!/usr/bin/env python3
"""
4h_Donchian20_Trend_EMA50_VolumeSpike_ATRStop_v1
Hypothesis: 4h Donchian(20) breakout filtered by 1d EMA50 trend and volume spike (>2.5x 30-period average).
Uses ATR(14) stoploss (2.5x) and discrete position sizing (0.30) to balance returns and fee drag.
Donchian breakouts capture momentum in both bull and bear markets, while 1d EMA50 filter avoids counter-trend trades.
Volume spike ensures institutional participation. Designed for low trade frequency (<50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    df_1d_close = df_1d['close'].values
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
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
    
    # === Volume filter: 30-period average (stricter) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # === Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Stricter volume filter: current volume > 2.5x 30-period average
            vol_filter = vol_current > 2.5 * vol_average
            
            # Long conditions: price > Donchian upper (breakout), 1d uptrend, volume filter
            long_breakout = price > highest_high[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < Donchian lower (breakdown), 1d downtrend, volume filter
            short_breakout = price < lowest_low[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic - stricter filters for fewer, higher-quality trades
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (wider 2.5x ATR to reduce premature exits)
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Donchian lower (breakdown)
            elif price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Check stoploss (wider 2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Donchian upper (breakout)
            elif price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Trend_EMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0