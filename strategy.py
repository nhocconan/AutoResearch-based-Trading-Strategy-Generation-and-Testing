#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
Works in both bull/bear via 1d ADX regime filter (trending vs ranging) and volume confirmation to avoid false signals.
Williams %R identifies overbought/oversold conditions for mean reversion entries.
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
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(low_1d)
    dx = np.zeros_like(high_1d)
    
    if atr[period] != 0:
        plus_di[period] = 100 * plus_dm[period] / atr[period]
        minus_di[period] = 100 * minus_dm[period] / atr[period]
        if plus_di[period] + minus_di[period] != 0:
            dx[period] = 100 * abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period])
    
    for i in range(period+1, len(high_1d)):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros_like(dx)
    adx[2*period] = np.mean(dx[period+1:2*period+1])
    for i in range(2*period+1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Williams %R for mean reversion
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    highest_high = np.zeros_like(high_12h)
    lowest_low = np.zeros_like(low_12h)
    
    for i in range(len(high_12h)):
        if i < 13:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high_12h[i-13:i+1])
            lowest_low[i] = np.min(low_12h[i-13:i+1])
    
    williams_r = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if highest_high[i] - lowest_low[i] != 0:
            williams_r[i] = -100 * (highest_high[i] - close_12h[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # neutral when no range
    
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # need ADX30, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending (avoid mean reversion in strong trends), ADX < 25 = ranging (good for mean reversion)
        ranging_market = adx_aligned[i] < 25
        
        # Volume filter: 12h volume > 1.5x 20-period MA (moderate to reduce trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND ranging market AND volume confirmation
            if williams_r_aligned[i] < -80 and ranging_market and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND ranging market AND volume confirmation
            elif williams_r_aligned[i] > -20 and ranging_market and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -30 for longs, -70 to -50 for shorts) or opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R returns from oversold
                if williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R returns from overbought
                if williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dADX_Regime_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0