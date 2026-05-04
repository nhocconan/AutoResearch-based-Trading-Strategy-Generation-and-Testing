#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 12h ADX Trend Filter and Volume Spike
# Long when Williams %R < -80 (oversold) AND 12h ADX > 25 (trending) AND volume > 2x 20 EMA
# Short when Williams %R > -20 (overbought) AND 12h ADX > 25 (trending) AND volume > 2x 20 EMA
# Exit when Williams %R returns to -50 level or ADX < 20 (weak trend)
# Uses 6h for entry timing, 12h for trend strength to avoid ranging markets.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-25 trades/year.
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_Extreme_12hADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Get 12h data for ADX trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # +DM = max(high - previous_high, 0) if > previous_low - low else 0
    dm1 = np.maximum(high_12h - np.roll(high_12h, 1), 0)
    dm2 = np.maximum(np.roll(low_12h, 1) - low_12h, 0)
    plus_dm = np.where((dm1 > dm2) & (dm1 > 0), dm1, 0)
    
    # -DM = max(previous_low - low, 0) if > high - previous_high else 0
    minus_dm = np.where((dm2 > dm1) & (dm2 > 0), dm2, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    
    # DX = |+DI - -DI| / (+DI + -DI) * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    # ADX = smoothed DX
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 12h ADX > 25 (trending) AND volume spike
            if (williams_r[i] < -80 and 
                adx_12h_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 12h ADX > 25 (trending) AND volume spike
            elif (williams_r[i] > -20 and 
                  adx_12h_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR ADX < 20 (weak trend)
            if (williams_r[i] > -50 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR ADX < 20 (weak trend)
            if (williams_r[i] < -50 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals