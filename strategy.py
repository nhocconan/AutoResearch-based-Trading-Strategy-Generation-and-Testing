#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. In trending markets (ADX>25 on 1d),
# we take trades in the direction of the trend: long when Bull Power > 0, short when Bear Power < 0.
# In ranging markets (ADX<=25), we fade extremes: long when Bear Power < -std, short when Bull Power > +std.
# Volume spike confirms conviction. Designed for 12-37 trades/year on 6h to work in both bull and bear markets.

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
    
    # Get 1d data for ADX regime filter
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
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_14 = np.concatenate([np.full(14, np.nan), tr_14])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]),
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]),
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = np.concatenate([np.full(14, np.nan), dm_plus_14])
    dm_minus_14 = np.concatenate([np.full(14, np.nan), dm_minus_14])
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14 = np.concatenate([np.full(27, np.nan), adx[13:]])  # ADX starts at index 27
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike filter (20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_13[i]) or np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_1d = adx_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Regime filter: ADX > 25 = trending, ADX <= 25 = ranging
            if adx_1d > 25:
                # Trending: trade with the trend
                if bp > 0 and vol_spike:  # Bull power positive
                    signals[i] = 0.25
                    position = 1
                elif br < 0 and vol_spike:  # Bear power negative
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging: fade extremes (mean reversion)
                # Calculate rolling std of bear power for entry threshold
                lookback = min(50, i+1)
                br_std = np.nanstd(bear_power[max(0, i-lookback+1):i+1])
                bp_std = np.nanstd(bull_power[max(0, i-lookback+1):i+1])
                
                if not (np.isnan(br_std) or np.isnan(bp_std)):
                    if br < -0.5 * br_std and vol_spike:  # Oversold
                        signals[i] = 0.25
                        position = 1
                    elif bp > 0.5 * bp_std and vol_spike:  # Overbought
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: bull power turns negative or loses volume confirmation
            if bp <= 0 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive or loses volume confirmation
            if br >= 0 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals