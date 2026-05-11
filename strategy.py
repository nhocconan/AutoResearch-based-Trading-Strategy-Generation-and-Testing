#!/usr/bin/env python3
"""
4h_Williams_R_Reversal_v1
Hypothesis: Williams %R identifies overbought/oversold conditions. In ranging markets,
extreme readings often precede mean-reverting moves. We combine with 1d ADX trend filter
to avoid counter-trend trades in strong trends, and use volume spike for confirmation.
Target: 100-180 trades over 4 years (25-45/year) on 4h timeframe.
"""

name = "4h_Williams_R_Reversal_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1D Data for ADX Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Avoid division by zero
        dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
        dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
        
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        dx = np.zeros_like(di_plus)
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1]) if len(dx) >= 2*period-1 else 0
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Williams %R Calculation (4h) ===
    def williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        
        for i in range(len(high)):
            if i < period - 1:
                highest_high[i] = np.max(high[:i+1])
                lowest_low[i] = np.min(low[:i+1])
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        wr = np.where((highest_high - lowest_low) != 0,
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # === Volume Spike Detection ===
    # Volume SMA(20) and check for 1.5x spike
    vol_sma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_sma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_sma[i] = np.mean(volume[i-19:i+1])
    
    volume_spike = volume > (vol_sma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 30)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Oversold (WR < -80) + volume spike + not strong uptrend (ADX < 25)
            if wr[i] < -80 and volume_spike[i] and adx_1d_aligned[i] < 25:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (WR > -20) + volume spike + not strong downtrend (ADX < 25)
            elif wr[i] > -20 and volume_spike[i] and adx_1d_aligned[i] < 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: WR returns above -50 (mean reversion complete) or strong trend emerges
            if wr[i] > -50 or adx_1d_aligned[i] >= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: WR returns below -50 or strong trend emerges
            if wr[i] < -50 or adx_1d_aligned[i] >= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals