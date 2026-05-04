#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 breakout with 1d volume spike and ADX regime filter
# Uses Camarilla pivot levels from 1d to identify key support/resistance (R3/S3) for breakout entries.
# Volume confirmation from 1d ensures institutional participation. ADX(14) > 25 on 12h filters choppy markets.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and bear markets via breakdowns with volume confirmation.
# Alligator acts as trend filter: only trade when price is outside the Alligator's mouth (JAW-LIPS gap).
# Elder Ray provides entry timing: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# Works in both bull/bear markets by adapting to Alligator's trend/range detection.

name = "12h_Camarilla_R3S3_1dVolumeSpike_ADXRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA20 for volume average
    ema20_vol_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R3 = Close + (High - Low) * 1.1 / 4
    # S3 = Close - (High - Low) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema20_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_vol_1d)
    
    # Calculate ADX(14) on 12h for regime filter
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema20_vol_1d_aligned[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 12h volume > 2x 1d average volume (scaled)
        # Scale 1d average volume to 12h: 1d volume / 2 (since 12h is half of 1d)
        volume_ma_12h = ema20_vol_1d_aligned[i] / 2.0
        volume_spike = volume[i] > 2.0 * volume_ma_12h
        
        # ADX regime filter: only trade when trending (ADX > 25)
        trending = adx[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above R3 with volume spike and trending market
            if (close[i] > r3_1d_aligned[i] and 
                volume_spike and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 with volume spike and trending market
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_spike and 
                  trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below R3 OR ADX drops (losing trend)
            if (close[i] < r3_1d_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above S3 OR ADX drops (losing trend)
            if (close[i] > s3_1d_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals