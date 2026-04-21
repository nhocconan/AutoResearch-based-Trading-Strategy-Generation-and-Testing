#!/usr/bin/env python3
"""
6h_ElderRay_WeeklyTrend_VolumeSpike_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) breakout filtered by weekly EMA50 trend and volume spike (>2.0x 30-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to balance returns and fee drag.
Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaw in choppy markets.
Designed to work in both bull and bear markets via weekly trend filter and volatility-adjusted exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for EMA50 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === Weekly OHLC for EMA50 trend filter ===
    df_1w_close = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema_50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: 30-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # === 6h EMA13 for Elder Ray calculation ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Volume filter: current volume > 2.0x 30-period average
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: Bull Power > 0 (bullish momentum), weekly uptrend, volume filter
            long_condition = bull_power[i] > 0
            long_trend = price > ema_50_1w_aligned[i]
            
            # Short conditions: Bear Power < 0 (bearish momentum), weekly downtrend, volume filter
            short_condition = bear_power[i] < 0
            short_trend = price < ema_50_1w_aligned[i]
            
            # Entry logic - balanced filters for moderate trade frequency
            if long_condition and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: Bear Power turns negative (momentum shift)
            elif bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: Bull Power turns positive (momentum shift)
            elif bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0