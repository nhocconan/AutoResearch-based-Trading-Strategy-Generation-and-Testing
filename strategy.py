#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter
# Uses 1d primary timeframe to target 30-100 trades over 4 years (7-25/year).
# Donchian channels from 1w provide strong support/resistance. Breakouts beyond upper/lower bands
# indicate momentum moves. Volume spike (2.0x 20-period average on 1w) confirms validity.
# 1w ADX > 25 filters for trending markets only, avoiding choppy conditions.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Works in bull via breakout longs,
# in bear via breakout shorts with trend filter.

name = "1d_Donchian20_Breakout_1wVolumeSpike_1wADX25_v1"
timeframe = "1d"
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
    
    # Calculate 1w Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    upper = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian levels to 1d timeframe (wait for completed 1w bar)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    
    # Volume confirmation: volume > 2.0x 20-period average on 1w
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_raw = df_1w['volume'].values > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_raw.astype(float))
    
    # Calculate 1w ADX(14) for trend filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = df_1w['high'].values - df_1w['low'].values
    tr2 = np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1))
    tr3 = np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where(
        (df_1w['high'].values - np.roll(df_1w['high'].values, 1)) > (np.roll(df_1w['low'].values, 1) - df_1w['low'].values),
        np.maximum(df_1w['high'].values - np.roll(df_1w['high'].values, 1), 0),
        0
    )
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where(
        (np.roll(df_1w['low'].values, 1) - df_1w['low'].values) > (df_1w['high'].values - np.roll(df_1w['high'].values, 1)),
        np.maximum(np.roll(df_1w['low'].values, 1) - df_1w['low'].values, 0),
        0
    )
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed +DM and -DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero when both DI are zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_upper = upper_aligned[i]
        curr_lower = lower_aligned[i]
        curr_volume_spike = volume_spike_aligned[i] > 0.5  # convert back to boolean
        curr_adx = adx_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish breakout: price breaks above upper band
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lower band
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below lower band (mean reversion)
            if curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above upper band (mean reversion)
            if curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals