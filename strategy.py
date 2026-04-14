# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and 1w ADX trend filter.
# Long when price breaks above 1d Donchian upper band (20-period) with 1w ADX > 25 and volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian lower band with 1w ADX > 25 and volume confirmation.
# Exit when price returns to the opposite Donchian band or ADX falls below 20.
# Designed to capture strong trending moves while avoiding false breakouts in low-volatility or ranging markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    lookback = 20
    donch_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Load 1w data ONCE for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX on 1w
    adx_period = 14
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_sum = pd.Series(tr).ewm(alpha=1/adx_period, adjust=False).mean().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/adx_period, adjust=False).mean().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/adx_period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/adx_period, adjust=False).mean().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_aligned = vol_ma  # Already calculated on lower timeframe
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 30)  # Need Donchian and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high AND strong trend
            if (close[i] > donch_high_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND strong trend
            elif (close[i] < donch_low_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or trend weakens (ADX < 20)
            if (close[i] <= donch_low_aligned[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian high or trend weakens (ADX < 20)
            if (close[i] >= donch_high_aligned[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian_1wADX_VolumeFilter_v2"
timeframe = "12h"
leverage = 1.0