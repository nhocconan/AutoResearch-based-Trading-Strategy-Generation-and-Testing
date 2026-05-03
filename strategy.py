#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; ADX>25 filters for trending markets
# Volume spike confirms institutional participation. Designed for low trade frequency
# (12-37/year) on 6h timeframe to minimize fee drag. Works in both bull and bear markets
# by taking trend-following entries only when higher timeframe trend is strong.

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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for trend strength filter
    # ADX calculation: +DM, -DM, TR, then smoothed
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    up_move = pd.Series(df_1d['high']).diff()
    down_move = pd.Series(df_1d['low']).diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 6h Elder Ray components
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bull power positive, ADX>25 (strong uptrend), volume spike
            if bull_power[i] > 0 and adx_aligned[i] > 25 and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative, ADX>25 (strong downtrend), volume spike
            elif bear_power[i] < 0 and adx_aligned[i] > 25 and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative OR ADX weakens (<20)
            if bull_power[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive OR ADX weakens (<20)
            if bear_power[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals