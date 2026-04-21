#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakout filtered by 1w EMA50 trend and 6h volume spike (2.0x average). 
Weekly EMA50 provides strong trend filter to avoid whipsaw in ranging markets. 
Volume confirmation ensures breakout validity. Discrete sizing (0.25) to manage drawdown.
Works in bull markets via breakout continuation and in bear markets via short breakdowns with trend filter.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper: highest high over past 20 bars (excluding current)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Donchian lower: lowest low over past 20 bars (excluding current)
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) 
            or np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        dc_upper = highest_high[i]
        dc_lower = lowest_low[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Only enter in direction of weekly trend
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > dc_upper) and (price > ema_trend) and volume_confirmed
            short_condition = (price < dc_lower) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price crosses below weekly EMA50)
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price crosses above weekly EMA50)
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0