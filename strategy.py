#!/usr/bin/env python3
"""
6h_12h_1d_VolumeWeighted_PivotBreakout
Hypothesis: Combines volume-weighted VWAP deviation with 12h trend filter and 1d pivot confluence.
Only takes long when price crosses above VWAP with volume expansion, price above 12h EMA50,
and near 1d S1 pivot (support). Short when price crosses below VWAP with volume expansion,
price below 12h EMA50, and near 1d R1 pivot (resistance).
VWAP mean reversion works in ranging markets while trend filter captures directional moves.
Pivot confluence adds institutional level validation. Designed for 6h timeframe to limit
trades to 20-40/year, reducing fee impact while maintaining edge in both bull/bear markets.
"""

name = "6h_12h_1d_VolumeWeighted_PivotBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Volume expansion: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (1.5 * vol_ma)
    
    # 1d data for pivot points (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate standard pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    df_1d['typical'] = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    pivot_p = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3.0
    pivot_r1 = 2 * pivot_p - df_1d['low'].shift(1)
    pivot_s1 = 2 * pivot_p - df_1d['high'].shift(1)
    
    # Align pivot points to 6h timeframe
    pivot_p_aligned = align_htf_to_ltf(prices, df_1d, pivot_p.values)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1d, pivot_r1.values)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1d, pivot_s1.values)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(vwap[i]) or
            np.isnan(pivot_p_aligned[i]) or
            np.isnan(pivot_r1_aligned[i]) or
            np.isnan(pivot_s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above VWAP with volume expansion,
            # price above 12h EMA50, and near S1 pivot (support)
            vwap_cross_up = close[i] > vwap[i] and close[i-1] <= vwap[i-1]
            near_s1 = abs(close[i] - pivot_s1_aligned[i]) / close[i] < 0.005  # within 0.5%
            
            if (vwap_cross_up and 
                volume_expansion[i] and 
                close[i] > ema_50_12h_aligned[i] and
                near_s1):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below VWAP with volume expansion,
            # price below 12h EMA50, and near R1 pivot (resistance)
            elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and
                  volume_expansion[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  abs(close[i] - pivot_r1_aligned[i]) / close[i] < 0.005):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below VWAP or moves far from S1
            vwap_cross_down = close[i] < vwap[i] and close[i-1] >= vwap[i-1]
            far_from_s1 = abs(close[i] - pivot_s1_aligned[i]) / close[i] > 0.015  # beyond 1.5%
            
            if vwap_cross_down or far_from_s1 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above VWAP or moves far from R1
            vwap_cross_up = close[i] > vwap[i] and close[i-1] <= vwap[i-1]
            far_from_r1 = abs(close[i] - pivot_r1_aligned[i]) / close[i] > 0.015  # beyond 1.5%
            
            if vwap_cross_up or far_from_r1 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals