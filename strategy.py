#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: Daily Camarilla pivot (R1/S1) breakout filtered by 1-week EMA50 trend and volume spike (>2.5x 30-period average).
Uses ATR(14) stoploss (2.5x) and discrete position sizing (0.30) to balance returns and fee drag.
Designed for low-frequency trading (target: 30-100 trades over 4 years) to minimize fee drag and improve test generalization.
Works in both bull and bear markets via 1-week trend filter and volatility-adjusted exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1-week for EMA50 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1-week OHLC for Camarilla pivot calculation (based on previous 1-week bar) ===
    df_1w_open = df_1w['open'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1-week bar
    range_1w = df_1w_high - df_1w_low
    r1_1w = df_1w_close + 0.275 * range_1w
    s1_1w = df_1w_close - 0.275 * range_1w
    
    # Align 1-week Camarilla levels to 1d timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 1-week EMA50 for trend filter ===
    ema_50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
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
            
            # Long conditions: price > R1 (breakout), 1-week uptrend, volume filter
            long_breakout = price > r1_1w_aligned[i]
            long_trend = price > ema_50_1w_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 1-week downtrend, volume filter
            short_breakout = price < s1_1w_aligned[i]
            short_trend = price < ema_50_1w_aligned[i]
            
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
            elif price < s1_1w_aligned[i]:
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
            elif price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0