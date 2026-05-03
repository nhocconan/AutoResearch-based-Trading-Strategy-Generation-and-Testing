#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 1d ADX > 25 ensures we only trade in trending markets (avoiding chop).
# Volume spike confirms institutional participation. Designed for 12-30 trades/year on 6h.
# Works in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends.

name = "6h_WilliamsR_Extreme_1dADX_Trend_Volume"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and ADX
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Williams %R(14) on 6h
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_14_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Williams %R using data up to current bar
        highest_high = np.max(high[i-13:i+1])  # 14-period high
        lowest_low = np.min(low[i-13:i+1])     # 14-period low
        williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        # Volume confirmation: 14-period EMA
        if i >= 13:
            vol_ema_14 = pd.Series(volume[i-13:i+1]).ewm(span=14, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_14 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_14)
        
        # Williams %R extreme conditions
        oversold = williams_r < -80
        overbought = williams_r > -20
        
        if position == 0:
            # Long: oversold in 1d uptrend (ADX > 25) with volume spike
            if oversold and adx_14_aligned[i] > 25 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: overbought in 1d downtrend (ADX > 25) with volume spike
            elif overbought and adx_14_aligned[i] > 25 and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 or loses 1d uptrend
            if williams_r > -50 or adx_14_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 or loses 1d downtrend
            if williams_r < -50 or adx_14_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals