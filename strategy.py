#!/usr/bin/env python3
"""
4h_PivotPoint_DeMarker_Squeeze_Breakout
Hypothesis: Combine daily Pivot Point levels with DeMarker overbought/oversold signals and Bollinger Band squeeze breakouts.
Long when price breaks above R1 with DeMarker oversold reversal and BB squeeze release, short when breaks below S1 with overbought reversal.
Uses 1-day ATR to filter low volatility chop. Designed for 4h to target 20-40 trades/year with high-conviction entries.
Works in bull markets via momentum breakouts and in bear via mean-reversion squeezes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_dema(close, period):
    """Double Exponential Moving Average"""
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2

def calculate_demark(high, low, close, period=13):
    """DeMarker indicator (0-1 scale)"""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    # DeMax: max(0, high - high_prev)
    demax = np.maximum(0, high[1:] - high[:-1])
    demax = np.concatenate([[np.nan], demax])
    
    # DeMin: max(0, low_prev - low)
    demin = np.maximum(0, low[:-1] - low[1:])
    demin = np.concatenate([demin, [np.nan]])
    
    # DeMarker = DeMax / (DeMax + DeMin)
    denom = demax + demin
    demark = np.where(denom != 0, demax / denom, np.nan)
    return demark

def calculate_bbands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = (upper - lower) / sma
    return upper, lower, width

def calculate_pivot(high, low, close):
    """Standard Pivot Point levels"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Pivot Points, DeMarker, ATR, and BBands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Pivot Points
    pivot_1d = np.zeros(len(df_1d))
    r1_1d = np.zeros(len(df_1d))
    s1_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        _, r1, _, s1, _ = calculate_pivot(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Calculate daily DeMarker (13-period)
    demark_1d = calculate_demark(high_1d, low_1d, close_1d, 13)
    
    # Calculate daily ATR (14-period)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate daily Bollinger Band width (20,2)
    _, _, bb_width_1d = calculate_bbands(close_1d, 20, 2.0)
    
    # Align all indicators to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    demark_1d_aligned = align_htf_to_ltf(prices, df_1d, demark_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(demark_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(bb_width_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 30-period average
        if i >= 30:
            vol_ma = prices['volume'].iloc[i-30:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility (chop) AND extreme volatility
        atr_percentile = np.percentile(atr_1d_aligned[max(0, i-49):i+1], 30) if i >= 30 else 0
        vol_filter = (atr_1d_aligned[i] > atr_percentile) and (atr_1d_aligned[i] < np.percentile(atr_1d_aligned[:i+1], 90) if i >= 30 else True)
        
        # Bollinger Band squeeze release: width expanding from low
        bb_squeeze_release = False
        if i >= 35:
            bb_width_min = np.min(bb_width_1d_aligned[i-15:i+1])
            bb_width_current = bb_width_1d_aligned[i]
            bb_squeeze_release = bb_width_current > bb_width_min * 1.2
        
        if position == 0:
            # Long: price breaks above R1 with DeMarker oversold reversal (<0.3) + volume + BB squeeze release
            if (price > r1_1d_aligned[i] and 
                demark_1d_aligned[i] < 0.3 and 
                volume_ok and 
                vol_filter and 
                bb_squeeze_release):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with DeMarker overbought (>0.7) + volume + BB squeeze release
            elif (price < s1_1d_aligned[i] and 
                  demark_1d_aligned[i] > 0.7 and 
                  volume_ok and 
                  vol_filter and 
                  bb_squeeze_release):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR DeMarker overbought OR volatility drops
            if (price < s1_1d_aligned[i] or 
                demark_1d_aligned[i] > 0.7 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR DeMarker oversold OR volatility drops
            if (price > r1_1d_aligned[i] or 
                demark_1d_aligned[i] < 0.3 or 
                not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_DeMarker_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0