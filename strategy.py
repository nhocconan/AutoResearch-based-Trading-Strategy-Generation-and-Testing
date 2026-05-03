#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Trend + Volume Confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (<-90 or >-10) 
# combined with 1d ADX > 25 (strong trend) and volume spike capture trend continuation 
# with controlled frequency. Designed for 12-30 trades/year on 6h to minimize fee drag 
# while working in both bull (trend continuation) and bear (extreme bounces) markets.

name = "6h_WilliamsR_Extreme_1dADX_VolumeSpike"
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
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    def williams_r(high, low, close, period):
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        # For each point, look back 'period' bars
        wr = np.full_like(close, np.nan)
        for i in range(period-1, len(close)):
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            if period_high != period_low:
                wr[i] = -100 * (period_high - close[i]) / (period_high - period_low)
            else:
                wr[i] = -50  # Avoid division by zero
        return wr
    
    wr_14 = williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(wr_14[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams %R extreme conditions
        wr_oversold = wr_14[i] < -90  # Extreme oversold
        wr_overbought = wr_14[i] > -10  # Extreme overbought
        
        # ADX trend condition (strong trend)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: extreme oversold + strong trend + volume spike
            if wr_oversold and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: extreme overbought + strong trend + volume spike
            elif wr_overbought and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns from extreme or trend weakens
            if wr_14[i] > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns from extreme or trend weakens
            if wr_14[i] < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals