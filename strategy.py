#!/usr/bin/env python3
# 6h_1d_ADX_Power_Momentum
# Hypothesis: Uses daily ADX for regime detection (trending vs ranging) and 6h Elder Ray power (bull/bear) for entry.
# In trending markets (ADX > 25), we take Elder Ray signals in direction of trend.
# In ranging markets (ADX < 20), we fade extreme Elder Ray readings.
# This adapts to market regimes and should work in both bull and bear markets.
# Target: 15-30 trades/year to minimize fee drag while capturing regime-appropriate moves.

name = "6h_1d_ADX_Power_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for ADX and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 1d ADX for trend strength ---
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = df_1d['high'] - df_1d['high'].shift(1)
    dm_minus = df_1d['low'].shift(1) - df_1d['low']
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth TR and DM
    atr = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # Calculate DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # --- 1d Elder Ray Power (Bull/Bear Power) ---
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = df_1d['close'].ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = df_1d['high'] - ema_13
    bear_power = df_1d['low'] - ema_13
    
    bull_power_values = bull_power.values
    bear_power_values = bear_power.values
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_values)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ADX (14+14=28) and EMA (13)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        if position == 0:
            # Trending market: ADX > 25
            if adx_val > 25:
                # Strong bull power and rising price -> long
                if bull_power_val > 0 and close[i] > close[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Strong bear power and falling price -> short
                elif bear_power_val < 0 and close[i] < close[i-1]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: ADX < 20
            elif adx_val < 20:
                # Extreme bull power -> fade (short)
                if bull_power_val > np.percentile(bull_power_aligned[max(0, i-50):i+1], 80):
                    signals[i] = -0.25
                    position = -1
                # Extreme bear power -> fade (long)
                elif bear_power_val < np.percentile(bear_power_aligned[max(0, i-50):i+1], 20):
                    signals[i] = 0.25
                    position = 1
        else:
            if position == 1:
                # Exit long: bull power turns negative OR ADX drops below 20 (trend weakening)
                if bull_power_val <= 0 or adx_val < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bear power turns positive OR ADX drops below 20 (trend weakening)
                if bear_power_val >= 0 or adx_val < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals