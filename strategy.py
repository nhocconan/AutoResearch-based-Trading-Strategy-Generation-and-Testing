#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_elder_ray_regime_v1
# Elder Ray (Bull/Bear power) with regime filter using ADX and 200 EMA.
# In trending markets (ADX > 25): go long when Bull Power > 0 and close > EMA200,
# short when Bear Power < 0 and close < EMA200.
# In ranging markets (ADX <= 25): fade extremes - long when Bear Power < -std,
# short when Bull Power > +std.
# Uses 1d Elder Ray for cleaner signals, aligned to 6x.
# Target: 15-30 trades/year per symbol for low friction and high win rate.
name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA200 on 1d close
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components and EMA200 to 6t
    bull_power = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema200 = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate ADX on 1d for regime detection (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Prepend first values
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volatility normalization for ranging signals
    # Use 20-period std of Bear/Bull power
    bull_power_series = pd.Series(bull_power)
    bear_power_series = pd.Series(bear_power)
    bull_std = bull_power_series.rolling(window=20, min_periods=20).std().values
    bear_std = bear_power_series.rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if not ready
        if np.isnan(ema200[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime: trending if ADX > 25, ranging if ADX <= 25
        is_trending = adx_aligned[i] > 25
        
        if is_trending:
            # Trending regime: follow Elder Ray signals with EMA200 filter
            if bull_power[i] > 0 and close[i] > ema200[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif bear_power[i] < 0 and close[i] < ema200[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit on opposite signal
            elif bear_power[i] < 0 and close[i] < ema200[i] and position == 1:
                position = 0
                signals[i] = 0.0
            elif bull_power[i] > 0 and close[i] > ema200[i] and position == -1:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Ranging regime: fade extremes
            # Long when Bear Power is significantly negative (oversold)
            if bear_power[i] < -bull_std[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short when Bull Power is significantly positive (overbought)
            elif bull_power[i] > bear_std[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit when power returns to neutral zone
            elif bear_power[i] > -0.5 * bull_std[i] and position == 1:
                position = 0
                signals[i] = 0.0
            elif bull_power[i] < 0.5 * bear_std[i] and position == -1:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals