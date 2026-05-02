#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX Trend Filter and Volume Spike
# Williams %R identifies overbought/oversold conditions; extreme readings below -80 or above -20
# with volume confirmation indicate potential reversals. 1d ADX > 25 ensures trades align with
# strong trending markets to avoid false signals in chop. Designed for 50-150 total trades
# over 4 years (12-37/year) on 6h timeframe. Works in bull markets (buying oversold in uptrend)
# and bear markets (selling overbought in downtrend) by only taking trades in direction of 1d ADX.

name = "6h_WilliamsR_Extreme_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0) if (high_1d[i] - high_1d[i-1]) > (low_1d[i-1] - low_1d[i]) else 0
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0) if (low_1d[i-1] - low_1d[i]) > (high_1d[i] - high_1d[i-1]) else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[1:period+1])  # Initial ATR
    
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 30, 50)  # Williams %R lookback, ADX warmup, buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R below -80 (oversold) with volume spike AND ADX > 25 (strong trend)
            if (williams_r[i] < -80 and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R above -20 (overbought) with volume spike AND ADX > 25 (strong trend)
            elif (williams_r[i] > -20 and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (exit oversold) OR ADX < 20 (trend weakening)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (exit overbought) OR ADX < 20 (trend weakening)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals