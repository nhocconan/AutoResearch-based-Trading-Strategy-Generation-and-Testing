#!/usr/bin/env python3
# 6h_VWAP_MeanReversion_With_Regime
# Hypothesis: Price reverts to VWAP during low volatility periods (choppy markets). 
# Uses 6h VWAP with 1d ADX regime filter: mean revert only when ADX < 20 (ranging market).
# Works in bull markets (mean reversion in rallies) and bear markets (mean reversion in declines) 
# by fading extreme moves from VWAP when trend is weak. Volume-weighted price acts as dynamic 
# support/resistance in sideways markets.

name = "6h_VWAP_MeanReversion_With_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        result[period-1] = np.nansum(x[:period])
        for i in range(period, len(x)):
            result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    adx_14_1d = adx
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate 6h VWAP (typical price * volume)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Distance from VWAP in ATR units (using 6-period ATR for normalization)
    def atr_6_period(high, low, close):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = np.zeros_like(close)
        for i in range(6, len(tr)):
            atr[i] = np.mean(tr[i-5:i+1])
        return atr
    
    atr_6 = atr_6_period(high, low, close)
    distance_from_vwap = (close - vwap) / np.where(atr_6 > 0, atr_6, np.inf)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily ADX (14+13), VWAP needs some data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(distance_from_vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only mean revert when ADX < 20 (ranging market)
        ranging = adx_14_1d_aligned[i] < 20
        
        if position == 0:
            # Long entry: price below VWAP + ranging market
            if ranging and distance_from_vwap[i] < -1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price above VWAP + ranging market
            elif ranging and distance_from_vwap[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above VWAP or trend strengthens
            if close[i] > vwap[i] or adx_14_1d_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below VWAP or trend strengthens
            if close[i] < vwap[i] or adx_14_1d_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals