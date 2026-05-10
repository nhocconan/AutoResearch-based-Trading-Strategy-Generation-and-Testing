#!/usr/bin/env python3
# 12h_Donchian20_1dTrend_VolumeSpike_v1
# Hypothesis: 12h breakout of daily Donchian(20) channels with 1d EMA50 trend filter and volume spike confirmation.
# Uses 1d trend for bias to avoid whipsaws in sideways markets, 12h for entry timing.
# Targets 20-40 trades/year to minimize fee drag. Works in bull/bear by trading breakouts aligned with higher timeframe trend.
# Added volume confirmation to reduce false breakouts.

name = "12h_Donchian20_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Daily high/low for Donchian channel (20-day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period rolling max/min for Donchian bands
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_up = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(donchian_up_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian upper band with uptrend and volume spike
            if (close[i] > donchian_up_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band with downtrend and volume spike
            elif (close[i] < donchian_low_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to Donchian midpoint or trend fails
            donchian_mid = (donchian_up_aligned[i] + donchian_low_aligned[i]) / 2
            
            if (close[i] < donchian_mid or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to Donchian midpoint or trend fails
            donchian_mid = (donchian_up_aligned[i] + donchian_low_aligned[i]) / 2
            
            if (close[i] > donchian_mid or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals