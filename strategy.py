#!/usr/bin/env python3
"""
6h_VolumeWeighted_RSI_Trend
Hypothesis: Combines volume-weighted RSI (VW-RSI) with 1-day trend filter to capture mean-reversion in ranging markets while avoiding trends. VW-RSI gives more weight to price moves on high volume, improving signal quality. Uses 1-day ADX to filter out trending markets (ADX > 25) where mean reversion fails. Targets 15-25 trades/year per symbol.
"""

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
    
    # RSI period
    rsi_period = 14
    # VW-RSI: weight price changes by volume
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_up = up * volume
    vol_down = down * volume
    
    # Smoothed volume-weighted RSI
    avg_vol_up = np.zeros(n)
    avg_vol_down = np.zeros(n)
    
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_vol_up[i] = np.mean(vol_up[i-rsi_period+1:i+1])
            avg_vol_down[i] = np.mean(vol_down[i-rsi_period+1:i+1])
        else:
            avg_vol_up[i] = (avg_vol_up[i-1] * (rsi_period-1) + vol_up[i]) / rsi_period
            avg_vol_down[i] = (avg_vol_down[i-1] * (rsi_period-1) + vol_down[i]) / rsi_period
    
    # Avoid division by zero
    rs = np.where(avg_vol_down != 0, avg_vol_up / avg_vol_down, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    adx_period = 14
    tr1 = np.zeros(len(high_1d))
    tr2 = np.zeros(len(high_1d))
    tr3 = np.zeros(len(high_1d))
    
    tr1[1:] = np.abs(high_1d[1:] - low_1d[1:])
    tr2[1:] = np.abs(high_1d[1:] - close_1d[:-1])
    tr3[1:] = np.abs(low_1d[1:] - close_1d[:-1])
    
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth TR, DM+
    tr_smooth = np.zeros(len(tr))
    dm_plus_smooth = np.zeros(len(dm_plus))
    dm_minus_smooth = np.zeros(len(dm_minus))
    
    for i in range(adx_period, len(tr)):
        if i == adx_period:
            tr_smooth[i] = np.sum(tr[i-adx_period+1:i+1])
            dm_plus_smooth[i] = np.sum(dm_plus[i-adx_period+1:i+1])
            dm_minus_smooth[i] = np.sum(dm_minus[i-adx_period+1:i+1])
        else:
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / adx_period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / adx_period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / adx_period) + dm_minus[i]
    
    # Avoid division by zero
    dm_plus_smooth = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    dm_minus_smooth = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    dx = np.where((dm_plus_smooth + dm_minus_smooth) != 0, 
                  100 * np.abs(dm_plus_smooth - dm_minus_smooth) / (dm_plus_smooth + dm_minus_smooth), 0)
    
    # Smooth DX to get ADX
    adx = np.zeros(len(dx))
    for i in range(adx_period, len(dx)):
        if i == adx_period:
            adx[i] = np.mean(dx[i-adx_period+1:i+1])
        else:
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # VW-RSI thresholds for mean reversion
    vw_rsi_oversold = 30
    vw_rsi_overbought = 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period, adx_period)
    
    for i in range(start_idx, n):
        if (np.isnan(vw_rsi[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (ADX < 25)
        if adx_aligned[i] > 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VW-RSI oversold
            if vw_rsi[i] < vw_rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short: VW-RSI overbought
            elif vw_rsi[i] > vw_rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: VW-RSI returns to neutral (50) or adverse move
            if vw_rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VW-RSI returns to neutral (50) or adverse move
            if vw_rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolumeWeighted_RSI_Trend"
timeframe = "6h"
leverage = 1.0