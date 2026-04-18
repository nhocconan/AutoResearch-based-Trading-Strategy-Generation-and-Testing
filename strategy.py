#!/usr/bin/env python3
"""
1d Bollinger Band Squeeze + Volume Spike + ADX Trend Filter
Hypothesis: Bollinger Band width identifies low volatility periods. A breakout from squeeze with volume confirmation and ADX > 20 captures the start of new trends in both bull and bear markets. Using daily timeframe reduces trade frequency to minimize fee drag while capturing major moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    if len(tr) < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def smooth_series(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = smooth_series(tr, period)
    plus_di = 100 * smooth_series(plus_dm, period) / np.where(atr != 0, atr, 1)
    minus_di = 100 * smooth_series(minus_dm, period) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_series(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Bollinger Bands on daily
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate BB middle (SMA)
    bb_mid = np.zeros_like(close)
    for i in range(n):
        if i < bb_length - 1:
            bb_mid[i] = np.mean(close[max(0, i-bb_length+1):i+1])
        else:
            bb_mid[i] = np.mean(close[i-bb_length+1:i+1])
    
    # Calculate standard deviation
    bb_std = np.zeros_like(close)
    for i in range(n):
        if i < bb_length - 1:
            bb_std[i] = np.std(close[max(0, i-bb_length+1):i+1])
        else:
            bb_std[i] = np.std(close[i-bb_length+1:i+1])
    
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # Bollinger Band squeeze: width below 20-period average
    bb_width_ma = np.zeros_like(bb_width)
    for i in range(n):
        if i < 19:
            bb_width_ma[i] = np.mean(bb_width[max(0, i-19):i+1])
        else:
            bb_width_ma[i] = np.mean(bb_width[i-19:i+1])
    bb_squeeze = bb_width < bb_width_ma * 0.8  # 20% below average width
    
    # Breakout: price closes outside Bollinger Bands
    breakout_up = close > bb_upper
    breakout_down = close < bb_lower
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(n):
        if i < 19:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # ADX trend filter from weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, period=14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(adx_1w_aligned[i]) or np.isnan(bb_width[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1w_aligned[i]
        squeeze_active = bb_squeeze[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: breakout up + volatility squeeze + volume spike + ADX > 20
            if (breakout_up[i] and 
                squeeze_active and 
                vol_ok and 
                adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Enter short: breakout down + volatility squeeze + volume spike + ADX > 20
            elif (breakout_down[i] and 
                  squeeze_active and 
                  vol_ok and 
                  adx_val > 20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle band or volatility expands significantly
            if close[i] < bb_mid[i] or bb_width[i] > bb_width_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle band or volatility expands significantly
            if close[i] > bb_mid[i] or bb_width[i] > bb_width_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Bollinger_Squeeze_VolumeSpike_ADXFilter"
timeframe = "1d"
leverage = 1.0