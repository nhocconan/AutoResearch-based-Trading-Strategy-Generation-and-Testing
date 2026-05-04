#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme + 1d ADX trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Uses 12h for primary signals (low trade frequency), 1d for ADX trend to avoid choppy markets.
# Discrete sizing (0.25) to balance return and fee drag. Target: 12-37 trades/year.
# Works in bull markets via longs in oversold conditions and bear markets via shorts in overbought conditions.
# Williams %R is effective in ranging markets which often precede trends, and ADX filter ensures we only trade in trending conditions.

name = "12h_WilliamsR_Extreme_1dADX_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Get 1d data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = pd.Series(high_1d - low_1d).values
    tr2 = pd.Series(np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]) )).values
    tr3 = pd.Series(np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]) )).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]) ).values
    down_move = pd.Series(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d).values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / (atr + 1e-10)
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Uptrend/Downtrend based on ADX > 25 and directional bias
    adx_strong = adx > 25
    uptrend_1d = adx_strong & (plus_di > minus_di)
    downtrend_1d = adx_strong & (minus_di > plus_di)
    
    # Align 1d ADX trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Williams %R alignment (already calculated on 12h data)
    williams_r_aligned = williams_r  # No need to align as it's already on 12h timeframe
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 1d uptrend AND volume spike
            if (williams_r_aligned[i] < -80 and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 1d downtrend AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (recovery from oversold) OR 1d trend changes to downtrend
            if (williams_r_aligned[i] > -50 or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (decline from overbought) OR 1d trend changes to uptrend
            if (williams_r_aligned[i] < -50 or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals