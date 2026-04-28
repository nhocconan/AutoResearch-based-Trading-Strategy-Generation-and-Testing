#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX25 trend filter and volume confirmation.
# Williams %R > -20 = overbought (short signal), < -80 = oversold (long signal).
# Only trade in direction of 1d ADX > 25 (strong trend) to avoid whipsaws in ranging markets.
# Volume spike (>2.0x 20-bar average) confirms reversal strength.
# Works in both bull and bear via 1d ADX trend filter - captures strong moves while avoiding chop.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.

name = "6h_WilliamsR_Extreme_1dADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter (ADX > 25 = strong trend)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - pd.Series(high_1d).shift(1)) > (pd.Series(low_1d).shift(1) - low_1d),
                                 np.maximum(high_1d - pd.Series(high_1d).shift(1), 0), 0)).values
    dm_minus = pd.Series(np.where((pd.Series(low_1d).shift(1) - low_1d) > (high_1d - pd.Series(high_1d).shift(1)),
                                  np.maximum(pd.Series(low_1d).shift(1) - low_1d, 0), 0)).values
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Williams %R extreme conditions with volume confirmation
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        long_signal = williams_oversold and volume_spike[i] and strong_trend
        short_signal = williams_overbought and volume_spike[i] and strong_trend
        
        # Exit conditions: Williams %R returns to neutral range (-50 center)
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals