#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop_v3
Hypothesis: 4h Camarilla pivot (R1/S1) breakout filtered by 12h EMA50 trend and volume spike (>2.5x 30-period average).
Uses ATR(14) stoploss (2.5x) and discrete position sizing (0.30) to balance returns and fee drag.
Stricter volume and trend filters reduce trade frequency for better test generalization while maintaining edge.
Designed to work in both bull and bear markets via 12h trend filter and volatility-adjusted exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA50 trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h OHLC for Camarilla pivot calculation (based on previous 12h bar) ===
    df_12h_open = df_12h['open'].values
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    range_12h = df_12h_high - df_12h_low
    r1_12h = df_12h_close + 0.275 * range_12h
    s1_12h = df_12h_close - 0.275 * range_12h
    
    # Align 12h Camarilla levels to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 12h EMA50 for trend filter ===
    ema_50_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
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
            
            # Long conditions: price > R1 (breakout), 12h uptrend, volume filter
            long_breakout = price > r1_12h_aligned[i]
            long_trend = price > ema_50_12h_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 12h downtrend, volume filter
            short_breakout = price < s1_12h_aligned[i]
            short_trend = price < ema_50_12h_aligned[i]
            
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
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Check stoploss (wider 2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop_v3"
timeframe = "4h"
leverage = 1.0