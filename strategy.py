#!/usr/bin/env python3
"""
6h_ADX_Trend_Filtered_Camarilla_Breakout
Hypothesis: Breakouts above/below Camarilla R3/S3 levels (from 1d high-low-close) are filtered by 1d ADX > 25 to ensure strong trending conditions, reducing false breakouts in choppy markets. Works in bull/bear by only taking breakouts in the direction of strong trends (ADX > 25 indicates trend strength regardless of direction).
"""

name = "6h_ADX_Trend_Filtered_Camarilla_Breakout"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from 1d data (using previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_upper = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_lower = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1d, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1d, camarilla_lower)
    
    # 1d ADX calculation (trend strength filter)
    # ADX requires +DI and -DI calculation
    # +DI = (Smoothed +DM / ATR) * 100
    # -DI = (Smoothed -DM / ATR) * 100
    # ADX = Smoothed |+DI - -DI| / (+DI + -DI) * 100
    
    # Calculate +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed values
    atr = np.zeros_like(tr)
    plus_di_smooth = np.zeros_like(plus_dm)
    minus_di_smooth = np.zeros_like(minus_dm)
    
    # Wilder's smoothing: first value is average, then smoothed
    atr[period-1] = np.mean(tr[:period])
    plus_di_smooth[period-1] = np.mean(plus_dm[:period])
    minus_di_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
        plus_di_smooth[i] = (1 - alpha) * plus_di_smooth[i-1] + alpha * plus_dm[i]
        minus_di_smooth[i] = (1 - alpha) * minus_di_smooth[i-1] + alpha * minus_dm[i]
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, (plus_di_smooth / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_di_smooth / atr) * 100, 0)
    
    # DX = |+DI - -DI| / (+DI + -DI) * 100
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 
                  0)
    
    # ADX = smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[:2*period-1])  # first ADX value
    for i in range(2*period, len(dx)):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter: >1.5x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 + ADX > 25 (strong trend) + volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + ADX > 25 (strong trend) + volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S3 (reversal level)
            if close[i] < camarilla_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R3 (reversal level)
            if close[i] > camarilla_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals