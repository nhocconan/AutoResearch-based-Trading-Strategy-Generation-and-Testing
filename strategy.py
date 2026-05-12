#!/usr/bin/env python3
name = "6h_ADX_DMI_Strength_Trend_Following"
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
    
    # Load daily data once for ADX/DMI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX and DMI on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    period = 14
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    plus_dm_smooth = np.sum(plus_dm[:period])
    minus_dm_smooth = np.sum(minus_dm[:period])
    
    for i in range(period, len(tr)):
        # Wilder's smoothing: new = old - (old/period) + current
        atr[i] = atr[i-1] - (atr[i-1]/period) + tr[i]
        plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth/period) + plus_dm[i]
        minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth/period) + minus_dm[i]
        
        # Calculate DI values
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth / atr[i]) * 100
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # Calculate DX and ADX
    dx = np.zeros_like(tr)
    adx = np.zeros_like(tr)
    
    for i in range(period, len(tr)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
        else:
            dx[i] = 0
    
    # ADX is smoothed DX
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX and DI to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Daily trend filter: EMA(50) for direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + +DI > -DI (bullish) + price > EMA50 + volume filter
            if (adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i] and
                close[i] > ema50_1d_aligned[i] and
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend) + -DI > +DI (bearish) + price < EMA50 + volume filter
            elif (adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakening (ADX < 20) or DI crossover bearish
            if (adx_aligned[i] < 20 or 
                minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakening (ADX < 20) or DI crossover bullish
            if (adx_aligned[i] < 20 or 
                plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals