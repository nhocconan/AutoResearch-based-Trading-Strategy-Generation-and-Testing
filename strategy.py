#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Donchian channels provide clear breakout levels. 1d ADX > 25 filters for trending markets only.
# Volume spike confirms breakout validity. Works in bull via breakout longs, in bear via breakout shorts.
# Discrete sizing 0.30 balances risk and minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dADX25_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Upper band: 20-day high
    upper_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-day low
    lower_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 4h timeframe (wait for completed 1d bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 1d ADX(14) for trend filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where(
        (df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
        np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0),
        0
    )
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where(
        (df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0),
        0
    )
    
    # Smoothed +DM and -DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish breakout: price breaks above upper band
                if curr_close > curr_upper:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lower band
                elif curr_close < curr_lower:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below lower band (mean reversion)
            if curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit when price rises above upper band (mean reversion)
            if curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals