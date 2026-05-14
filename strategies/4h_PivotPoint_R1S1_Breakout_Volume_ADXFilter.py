#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Pivot Point and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Pivot Point and support/resistance levels
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*Pivot - Low, S1 = 2*Pivot - High
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align to 4h (previous day's levels available at 4h open)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d ADX for trend strength filter
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    tr_smooth = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial values
    tr_smooth[atr_period-1] = np.mean(tr[:atr_period])
    plus_dm_smooth[atr_period-1] = np.mean(plus_dm[:atr_period])
    minus_dm_smooth[atr_period-1] = np.mean(minus_dm[:atr_period])
    
    # Wilder's smoothing
    for i in range(atr_period, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / atr_period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / atr_period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / atr_period) + minus_dm[i]
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = np.zeros_like(plus_di)
    dx_mask = (plus_di + minus_di) > 0
    dx[dx_mask] = 100 * np.abs(plus_di[dx_mask] - minus_di[dx_mask]) / (plus_di[dx_mask] + minus_di[dx_mask])
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[2*atr_period-1] = np.mean(dx[atr_period:2*atr_period])
    for i in range(2*atr_period, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / atr_period) + dx[i]
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 6-period average (1.5 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: break above R1 with volume and strong trend (ADX > 25)
            if price > r1_1d_aligned[i] and adx_1d_aligned[i] > 25 and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 with volume and strong trend (ADX > 25)
            elif price < s1_1d_aligned[i] and adx_1d_aligned[i] > 25 and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below S1 or trend weakens (ADX < 20)
            if price < s1_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or trend weakens (ADX < 20)
            if price > r1_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0