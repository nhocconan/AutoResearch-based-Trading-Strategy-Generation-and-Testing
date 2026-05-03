#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX25 regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; strong bull/bear power with ADX>25
# indicates trending markets suitable for continuation trades. Volume spike confirms institutional
# participation. Designed for low trade frequency (12-37/year) on 6h to minimize fee drag.
# Works in both bull and bear markets by trading with the trend as measured by ADX regime.

name = "6h_ElderRay_1dADX25_VolumeSpike_Regime"
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
    
    # Get 1d data for Elder Ray and ADX calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Calculate 1d ADX for regime filter
    # True Range
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)
    tr3 = np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = df_1d['high'].values - df_1d['high'].shift(1).values
    down_move = df_1d['low'].shift(1).values - df_1d['low'].values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and ADX calculation
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: strong bull power (buying pressure) in uptrend regime with volume spike
            if bull_power_aligned[i] > 0 and adx_aligned[i] > 25 and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: strong bear power (selling pressure) in downtrend regime with volume spike
            elif bear_power_aligned[i] < 0 and adx_aligned[i] > 25 and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power weakens or regime changes
            if bull_power_aligned[i] <= 0 or adx_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power weakens or regime changes
            if bear_power_aligned[i] >= 0 or adx_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals