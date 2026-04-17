#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + 1d ADX Trend + Volume Spike.
Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (strong trend) AND volume > 2.0x average.
Short when Williams %R > -20 (overbought) AND 1d ADX > 25 AND volume > 2.0x average.
Exit when Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakens).
Uses 1d for ADX trend filter, 4h for price/Williams %R/volume.
Target: 50-150 total trades over 4 years (12-37/year). Works in both bull (trend continuation) and bear (oversold/overbought bounces).
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(close)
        dm_minus = np.zeros_like(close)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        atr = np.zeros_like(close)
        dmplus_smooth = np.zeros_like(close)
        dmminus_smooth = np.zeros_like(close)
        
        # Initial values (simple average)
        atr[period] = np.mean(tr[1:period+1])
        dmplus_smooth[period] = np.mean(dm_plus[1:period+1])
        dmminus_smooth[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dmplus_smooth[i] = (dmplus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dmminus_smooth[i] = (dmminus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(close)
        di_minus = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr[i] > 0:
                di_plus[i] = 100 * dmplus_smooth[i] / atr[i]
                di_minus[i] = 100 * dmminus_smooth[i] / atr[i]
            else:
                di_plus[i] = 0
                di_minus[i] = 0
        
        # DX and ADX
        dx = np.zeros_like(close)
        for i in range(period, len(close)):
            if di_plus[i] + di_minus[i] > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
            else:
                dx[i] = 0
        
        adx = np.zeros_like(close)
        # Initial ADX (simple average of first period DX values)
        if len(dx) >= 2*period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            # Wilder's smoothing for ADX
            for i in range(2*period, len(close)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 4h Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        williams_r = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if highest_high[i] - lowest_low[i] != 0:
                williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
            else:
                williams_r[i] = -50  # neutral
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike (current volume > 2.0x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        williams_val = williams_r[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: ADX > 25 = strong trend
        is_strong_trend = adx_val > 25
        # Weak trend filter: ADX < 20 = trend weakening
        is_weak_trend = adx_val < 20
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND strong trend AND volume spike
            if williams_val < -80 and is_strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND strong trend AND volume spike
            elif williams_val > -20 and is_strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR trend weakens
            if williams_val >= -50 or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR trend weakens
            if williams_val <= -50 or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_ADXTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0