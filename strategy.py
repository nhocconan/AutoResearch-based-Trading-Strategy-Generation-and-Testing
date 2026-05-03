#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter with volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; strong readings indicate momentum.
# 1d ADX > 25 filters for trending markets to avoid whipsaws in ranges.
# Volume spike confirms institutional participation. Designed for low trade frequency
# (12-37/year) on 6h timeframe to minimize fee drag. Works in both bull and bear markets
# by trading with the trend when momentum is confirmed.

name = "6h_ElderRay_1dADX_VolumeSpike_Regime"
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
    
    # Get 1d data for ADX and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / atr.replace(0, np.nan))
    
    # DX and ADX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_strong = adx_values > 25  # Strong trend threshold
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bull Power: high - EMA13 > 0 and strong
            bull_power = high[i] - ema_13_aligned[i]
            # Bear Power: EMA13 - low > 0 and strong
            bear_power = ema_13_aligned[i] - low[i]
            
            # Long: strong bull power + ADX strong trend + volume spike
            if bull_power > 0 and bear_power < bull_power and adx_strong_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: strong bear power + ADX strong trend + volume spike
            elif bear_power > 0 and bull_power < bear_power and adx_strong_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bear power exceeds bull power or ADX weakens
            bull_power = high[i] - ema_13_aligned[i]
            bear_power = ema_13_aligned[i] - low[i]
            if bear_power >= bull_power or not adx_strong_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power exceeds bear power or ADX weakens
            bull_power = high[i] - ema_13_aligned[i]
            bear_power = ema_13_aligned[i] - low[i]
            if bull_power >= bear_power or not adx_strong_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals