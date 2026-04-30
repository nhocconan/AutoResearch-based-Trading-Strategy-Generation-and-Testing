#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter
# Uses 1d primary timeframe to target 30-100 trades over 4 years (7-25/year).
# Donchian channels from 1w provide strong weekly support/resistance. Breakouts beyond
# upper/lower bands indicate momentum moves. Volume spike (2.0x 20-period average) confirms validity.
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
    
    # Calculate 1w Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Donchian upper/lower bands from previous 20 completed 1w bars
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1w Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: volume > 2.0x 20-period average on 1w
    vol_ma_20_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike_1w = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w * 2.0 <= df_1w['volume'].values)
    
    # Calculate 1w ADX(14) for trend filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr1.iloc[0] = 0  # first bar has no previous close
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values
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
    dm_plus.iloc[0] = 0
    dm_minus.iloc[0] = 0
    
    # Smoothed +DM and -DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero when both DI are zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_1w[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = donchian_high_aligned[i]
        curr_low = donchian_low_aligned[i]
        curr_volume_spike = volume_spike_1w[i]
        curr_adx = adx_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish breakout: price breaks above Donchian high
                if curr_close > curr_high:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Donchian low
                elif curr_close < curr_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price drops below Donchian low (mean reversion)
            if curr_close < curr_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high (mean reversion)
            if curr_close > curr_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals