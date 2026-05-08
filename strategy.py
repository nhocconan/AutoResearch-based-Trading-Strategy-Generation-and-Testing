#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_Slope_Crossover_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h high, low, close
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range (TR) calculation for 12h
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement (DM) calculation for 12h
    up_move = high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])
    down_move = np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed arrays
    tr14 = np.zeros_like(tr)
    plus_dm14 = np.zeros_like(plus_dm)
    minus_dm14 = np.zeros_like(minus_dm)
    
    # First value is simple average
    tr14[period-1] = np.mean(tr[:period])
    plus_dm14[period-1] = np.mean(plus_dm[:period])
    minus_dm14[period-1] = np.mean(minus_dm[:period])
    
    # Wilder's smoothing for rest
    for i in range(period, len(tr)):
        tr14[i] = tr14[i-1] - (tr14[i-1] / period) + tr[i]
        plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / period) + plus_dm[i]
        minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / period) + minus_dm[i]
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = np.zeros_like(dx)
    
    # First ADX value is simple average of DX
    adx[2*period-2] = np.mean(dx[period-1:2*period-1])
    
    # Wilder's smoothing for ADX
    for i in range(2*period-1, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # ADX slope calculation (3-period slope)
    adx_slope = np.zeros_like(adx)
    adx_slope[3:] = (adx[3:] - adx[:-3]) / 3
    
    # Align ADX and slope to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    adx_slope_aligned = align_htf_to_ltf(prices, df_12h, adx_slope)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for ADX calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(adx_slope_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX > 20 (trending), ADX slope positive, price above EMA50, volume filter
            long_cond = (adx_aligned[i] > 20) and (adx_slope_aligned[i] > 0) and (close[i] > ema_50_12h_aligned[i]) and volume_filter[i]
            # Short conditions: ADX > 20 (trending), ADX slope negative, price below EMA50, volume filter
            short_cond = (adx_aligned[i] > 20) and (adx_slope_aligned[i] < 0) and (close[i] < ema_50_12h_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX falls below 20 or slope turns negative
            if (adx_aligned[i] < 20) or (adx_slope_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX falls below 20 or slope turns positive
            if (adx_aligned[i] < 20) or (adx_slope_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals