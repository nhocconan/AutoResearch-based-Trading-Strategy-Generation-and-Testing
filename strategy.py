#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with 1d volume confirmation and 1d ADX trend filter.
# Uses 12h Donchian channels for breakout signals, confirmed by 1d volume spikes and 1d ADX > 25.
# Long when price breaks above 12h upper band with volume spike and ADX > 25.
# Short when price breaks below 12h lower band with volume spike and ADX > 25.
# Exit when price returns to 12h middle band (mean reversion) or ADX drops below 20.
# Designed for low trade frequency (20-40/year) to avoid fee drag. Combines trend breakout with volume and trend strength confirmation.

name = "4h_12hDonchian_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume EMA (20-period)
    vol_ema = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ema_aligned = align_htf_to_ltf(prices, df_1d, vol_ema)
    
    # Volume spike: current 4h volume > 2x 1d volume EMA (aligned)
    vol_spike = volume > (vol_ema_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 12h upper band + volume spike + ADX > 25
            if close[i] > donchian_high_aligned[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h lower band + volume spike + ADX > 25
            elif close[i] < donchian_low_aligned[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h middle band OR ADX drops below 20
            if close[i] > donchian_mid_aligned[i] and close[i] < donchian_high_aligned[i]:
                # Only exit if we're in the upper half (mean reversion to middle)
                if close[i] < (donchian_high_aligned[i] + donchian_mid_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h middle band OR ADX drops below 20
            if close[i] < donchian_low_aligned[i] and close[i] > donchian_high_aligned[i]:
                # Only exit if we're in the lower half (mean reversion to middle)
                if close[i] > (donchian_low_aligned[i] + donchian_mid_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals