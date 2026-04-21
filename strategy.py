#!/usr/bin/env python3
"""
1d_ChaikinMoneyFlow_Reversal
Hypothesis: On daily timeframe, Chaikin Money Flow (CMF) below -0.25 indicates distribution (sell pressure), above +0.25 indicates accumulation (buy pressure). Combined with price > 200-day SMA for long bias and price < 200-day SMA for short bias, this captures institutional flow extremes. Weekly ADX > 25 filters for trending conditions to avoid whipsaws in ranging markets. Works in bull/bear by taking reversal signals aligned with higher timeframe trend. Targets 10-25 trades/year with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_cmf(high, low, close, volume, period=20):
    """Calculate Chaikin Money Flow (CMF)"""
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)  # Avoid division by zero
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # CMF = sum(MFV, period) / sum(volume, period)
    mfv_sum = np.zeros_like(mfv)
    vol_sum = np.zeros_like(volume)
    
    for i in range(len(mfv)):
        if i < period:
            mfv_sum[i] = np.sum(mfv[max(0, i-period+1):i+1])
            vol_sum[i] = np.sum(volume[max(0, i-period+1):i+1])
        else:
            mfv_sum[i] = mfv_sum[i-1] + mfv[i] - mfv[i-period]
            vol_sum[i] = vol_sum[i-1] + volume[i] - volume[i-period]
    
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    return cmf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth using Wilder's smoothing
    tr_period = np.zeros_like(tr)
    dm_plus_period = np.zeros_like(dm_plus)
    dm_minus_period = np.zeros_like(dm_minus)
    
    tr_period[0] = tr[0]
    dm_plus_period[0] = dm_plus[0]
    dm_minus_period[0] = dm_minus[0]
    
    for i in range(1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
        dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
        dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    adx = np.zeros_like(dx)
    if len(dx) >= period:
        adx[period-1] = np.mean(dx[:period])  # First ADX value
    
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate daily CMF
    high_d = prices['high'].values
    low_d = prices['low'].values
    close_d = prices['close'].values
    volume_d = prices['volume'].values
    cmf = calculate_cmf(high_d, low_d, close_d, volume_d, 20)
    
    # Calculate 200-day SMA for bias filter
    close_series = pd.Series(close_d)
    sma_200 = close_series.rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after 200-day SMA is ready
        # Skip if weekly ADX not ready
        if np.isnan(adx_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trending = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: CMF > 0.25 (accumulation) + price above 200-day SMA + weekly trending
            if cmf[i] > 0.25 and close_d[i] > sma_200[i] and trending:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.25 (distribution) + price below 200-day SMA + weekly trending
            elif cmf[i] < -0.25 and close_d[i] < sma_200[i] and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF drops below 0 (distribution begins) or weekly ADX < 20 (losing trend)
            if cmf[i] < 0 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF rises above 0 (accumulation begins) or weekly ADX < 20 (losing trend)
            if cmf[i] > 0 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_ChaikinMoneyFlow_Reversal"
timeframe = "1d"
leverage = 1.0