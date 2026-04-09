#!/usr/bin/env python3
# 6h_adx_di_volume_v1
# Hypothesis: 6h strategy using ADX and DI crossover for trend strength/direction from 1d timeframe, with volume confirmation on 6h.
# Long: +DI > -DI (bullish) + ADX > 25 (strong trend) + volume spike
# Short: -DI > +DI (bearish) + ADX > 25 (strong trend) + volume spike
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.
# Works in bull/bear by only taking strong trend trades, avoiding chop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_di_volume_v1"
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
    
    # 1d HTF data for ADX/DI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement (+DM and -DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing, alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed TR, +DM, -DM
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # First values (simple average)
    atr[period-1] = np.mean(tr[:period])
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    # Wilder's smoothing
    for i in range(period, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
        plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + plus_dm[i] * alpha
        minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + minus_dm[i] * alpha
    
    # Avoid division by zero
    atr[atr == 0] = 1e-10
    
    # DI values
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero when both DI are zero
    dx[plus_di + minus_di == 0] = 0
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period-1])  # First ADX value
    for i in range(2*period, len(dx)):
        adx[i] = adx[i-1] * (1 - alpha) + dx[i] * alpha
    
    # Align to 6h timeframe
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX weakens (<20) OR DI crossover bearish
            if adx_aligned[i] < 20 or minus_di_aligned[i] > plus_di_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX weakens (<20) OR DI crossover bullish
            if adx_aligned[i] < 20 or plus_di_aligned[i] > minus_di_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and strong trend
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            strong_trend = adx_aligned[i] > 25
            
            if volume_confirmed and strong_trend:
                # Long: +DI > -DI (bullish)
                if plus_di_aligned[i] > minus_di_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: -DI > +DI (bearish)
                elif minus_di_aligned[i] > plus_di_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals