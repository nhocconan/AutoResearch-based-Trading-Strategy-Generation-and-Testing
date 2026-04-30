#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter
# Donchian channels from 1d provide strong support/resistance levels. Breakouts beyond upper/lower bands indicate continuation with momentum.
# 1w volume spike (2.0x 20-period average) confirms breakout validity. 1w ADX > 20 filters for trending markets only on weekly timeframe.
# Works in bull via breakout longs, in bear via breakout shorts. Discrete sizing 0.25 balances risk and minimizes fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wVolumeSpike_1wADX20_v1"
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
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w volume spike (volume > 2.0x 20-period average)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = df_1w['volume'].values > (2.0 * vol_ma_20_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    # Calculate 1w ADX(14) for trend filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where(
        (df_1w['high'] - df_1w['high'].shift(1)) > (df_1w['low'].shift(1) - df_1w['low']),
        np.maximum(df_1w['high'] - df_1w['high'].shift(1), 0),
        0
    )
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
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
    
    # Align 1w ADX to 1d timeframe (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_adx = adx_1w_aligned[i]
        curr_volume_spike = volume_spike_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (ADX > 20)
            if curr_volume_spike and curr_adx > 20:
                # Bullish breakout: price breaks above upper Donchian band
                if curr_close > curr_high_20:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lower Donchian band
                elif curr_close < curr_low_20:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below lower Donchian band (mean reversion)
            if curr_close < curr_low_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above upper Donchian band (mean reversion)
            if curr_close > curr_high_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals