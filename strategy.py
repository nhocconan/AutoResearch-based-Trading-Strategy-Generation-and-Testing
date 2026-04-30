#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX > 25 trend filter
# Donchian channel breakouts capture strong momentum moves. Volume spike (2.0x 20-period average) confirms breakout validity.
# 1w ADX > 25 ensures we only trade in strong trending markets, reducing whipsaws in ranging conditions.
# Works in bull via breakout longs, in bear via breakout shorts. Discrete sizing 0.25 minimizes fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_Donchian20_1wVolumeSpike_1wADX25_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = vol_1w > (2.0 * vol_ma_20_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    # Calculate 1w ADX(14) for trend filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM and -DM
    dm_plus = np.where(
        (df_1w['high'] - df_1w['high'].shift(1)) > (df_1w['low'].shift(1) - df_1w['low']),
        np.maximum(df_1w['high'] - df_1w['high'].shift(1), 0),
        0
    )
    dm_minus = np.where(
        (df_1w['low'].shift(1) - df_1w['low']) > (df_1w['high'] - df_1w['high'].shift(1)),
        np.maximum(df_1w['low'].shift(1) - df_1w['low'], 0),
        0
    )
    
    # Smoothed +DM and -DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(volume_spike_1w_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high_roll = high_roll[i]
        curr_low_roll = low_roll[i]
        curr_volume_spike = volume_spike_1w_aligned[i]
        curr_adx = adx_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and strong trending market (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish breakout: price breaks above upper Donchian channel
                if curr_close > curr_high_roll:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lower Donchian channel
                elif curr_close < curr_low_roll:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below the midpoint of the Donchian channel
            midpoint = (curr_high_roll + curr_low_roll) / 2.0
            if curr_close < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above the midpoint of the Donchian channel
            midpoint = (curr_high_roll + curr_low_roll) / 2.0
            if curr_close > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals