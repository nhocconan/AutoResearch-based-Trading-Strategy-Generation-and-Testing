# 2025-06-08: 12h_1d_camarilla_breakout_adx_v1
# Hypothesis: 12h Camarilla breakout with 1d volume spike and ADX trend filter
# Works in bull/bear: breakouts capture trends, volume confirms institutional interest,
# ADX filter avoids whipsaws in ranging markets. Target: 12-37 trades/year.
# Timeframe: 12h (primary), 1d (HTF)
# Expected trades: 50-150 total over 4 years.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar data (avoid look-ahead)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate 1d Camarilla H3/L3 levels
    pivot_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d_prev = high_1d_prev - low_1d_prev
    h3_prev = pivot_prev + (range_1d_prev * 1.1 / 4)
    l3_prev = pivot_prev - (range_1d_prev * 1.1 / 4)
    
    # Align to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    
    # 1d volume spike: volume > 2.5x 20-day average (stricter to reduce trades)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'] > (vol_ma_1d * 2.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # ADX trend filter on 12h (avoid ranging markets)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros(n)
    dm_plus_smooth = np.zeros(n)
    dm_minus_smooth = np.zeros(n)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI and DX
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    # ADX: smoothed DX
    adx = np.zeros(n)
    adx[2*period-1] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # ADX filter: trend present when ADX > 25
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above H3 with volume spike and ADX trend filter
        long_signal = close[i] > h3_aligned[i] and vol_spike_aligned[i] and adx_filter[i]
        # Short: break below L3 with volume spike and ADX filter
        short_signal = close[i] < l3_aligned[i] and vol_spike_aligned[i] and adx_filter[i]
        
        # Exit when price returns to pivot level
        pivot_prev_val = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev_val)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals