#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d Volume Spike + ADX Regime Filter
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In trending markets (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# In ranging markets (ADX < 20): fade extremes - long when Bear Power < -std and rising, short when Bull Power > std and falling
# Volume confirmation (1d avg volume spike) ensures institutional participation
# Works in bull/bear: ADX regime filter adapts, Elder Ray captures momentum/mean reversion appropriately
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_elder_ray_volume_adx_v1"
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
    
    # Load 1d data ONCE before loop for volume, EMA13, and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Calculate 1d ADX (14-period) for regime filtering
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate rolling statistics for Elder Ray thresholds (using 1d aligned data)
    bull_power_ma = pd.Series(bull_power_1d_aligned).rolling(window=20, min_periods=20).mean().values
    bull_power_std = pd.Series(bull_power_1d_aligned).rolling(window=20, min_periods=20).std().values
    bear_power_ma = pd.Series(bear_power_1d_aligned).rolling(window=20, min_periods=20).mean().values
    bear_power_std = pd.Series(bear_power_1d_aligned).rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(bull_power_ma[i]) or np.isnan(bull_power_std[i]) or
            np.isnan(bear_power_ma[i]) or np.isnan(bear_power_std[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 1d average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume_1d_aligned[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        bull_ma = bull_power_ma[i]
        bull_std = bull_power_std[i]
        bear_ma = bear_power_ma[i]
        bear_std = bear_power_std[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when bull power turns negative
                if bull_power <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when bull power returns to mean
                if bull_power >= bull_ma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when bear power turns positive
                if bear_power >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when bear power returns to mean
                if bear_power <= bear_ma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx > 25:  # Trending regime - follow momentum
                # Long when bull power is positive and rising above zero
                if bull_power > 0 and bull_power > bull_power_1d_aligned[max(0, i-1)]:
                    position = 1
                    signals[i] = 0.25
                # Short when bear power is negative and falling below zero
                elif bear_power < 0 and bear_power < bear_power_1d_aligned[max(0, i-1)]:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime - mean reversion
                # Long when bear power is significantly below mean and rising
                if bear_power < (bear_ma - 0.5 * bear_std) and bear_power > bear_power_1d_aligned[max(0, i-1)]:
                    position = 1
                    signals[i] = 0.25
                # Short when bull power is significantly above mean and falling
                elif bull_power > (bull_ma + 0.5 * bull_std) and bull_power < bull_power_1d_aligned[max(0, i-1)]:
                    position = -1
                    signals[i] = -0.25
    
    return signals