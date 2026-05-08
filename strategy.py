#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX(14) + Volume Spike + 1w EMA20 Trend Filter
# Long when ADX > 25 (trending) + volume > 2x 20-period avg + price above 1w EMA20
# Short when ADX > 25 + volume > 2x 20-period avg + price below 1w EMA20
# Exit when ADX < 20 (range) or price crosses 1w EMA20 in opposite direction
# Targets 15-25 trades per year for low fee drag (< 100 total over 4 years)
# ADX filters choppy markets, volume confirms momentum, 1w EMA provides multi-timeframe trend bias

name = "1d_ADX14_VolumeSpike_1wEMA20_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Directional movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr_smooth = smooth_wilder(tr, period)
    plus_dm_smooth = smooth_wilder(plus_dm, period)
    minus_dm_smooth = smooth_wilder(minus_dm, period)
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, period)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        adx_val = adx[i]
        ema20_val = ema20_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: ADX > 25 (trending) + volume spike + price above 1w EMA20
            if adx_val > 25 and vol_spike_val and close_val > ema20_val:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (trending) + volume spike + price below 1w EMA20
            elif adx_val > 25 and vol_spike_val and close_val < ema20_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 (range) or price crosses below 1w EMA20
            if adx_val < 20 or close_val < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 (range) or price crosses above 1w EMA20
            if adx_val < 20 or close_val > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals