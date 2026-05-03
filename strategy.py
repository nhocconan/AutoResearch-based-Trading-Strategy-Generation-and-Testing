#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions; extreme readings (< -90 or > -10) often precede reversals.
# 1d ADX > 25 confirms a strong daily trend to avoid counter-trend trades in choppy markets.
# Volume confirmation ensures breakouts have participation. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via oversold bounces in uptrends and in bear markets via overbought reversals in downtrends.

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = WilderSmooth(tr, 25)
    plus_di_1d = 100 * WilderSmooth(plus_dm, 25) / atr_1d
    minus_di_1d = 100 * WilderSmooth(minus_dm, 25) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = WilderSmooth(dx_1d, 25)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R on 6h timeframe (14-period)
    def calculate_williams_r(high, low, close, period):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_6h = calculate_williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA and WR
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr_6h[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: Williams %R oversold (< -90) in uptrend alignment with volume spike
            if wr_6h[i] < -90 and adx_1d_aligned[i] > 25 and plus_di_1d_aligned[i] > minus_di_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -10) in downtrend alignment with volume spike
            elif wr_6h[i] > -10 and adx_1d_aligned[i] > 25 and minus_di_1d_aligned[i] > plus_di_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or loses uptrend alignment
            if wr_6h[i] > -50 or plus_di_1d_aligned[i] <= minus_di_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or loses downtrend alignment
            if wr_6h[i] < -50 or minus_di_1d_aligned[i] <= plus_di_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals