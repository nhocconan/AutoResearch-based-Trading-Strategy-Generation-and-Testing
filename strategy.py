#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# Long when price breaks above Donchian upper(20) AND volume > 2.0x 20-period average AND 1w ADX > 25 (strong trend)
# Short when price breaks below Donchian lower(20) AND volume > 2.0x 20-period average AND 1w ADX > 25 (strong trend)
# Exit when price crosses back to Donchian midpoint OR 1w ADX < 20 (weak trend)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian channels provide structure, volume spike confirms participation, 1w ADX filters for trending regimes
# to avoid whipsaws in ranging markets. Works in bull/bear via breakouts in strong trends.

name = "12h_Donchian20_VolumeSpike_1wADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h data (using 20-period lookback)
    # Upper = max(high, 20), Lower = min(low, 20), Midpoint = (Upper + Lower)/2
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate ADX on 1w data for trend strength filter
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) >= 30:
        # Calculate True Range (TR)
        tr1 = np.abs(high_1w[1:] - low_1w[1:])
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # First TR is NaN
        
        # Calculate +DM and -DM
        up_move = np.concatenate([[np.nan], high_1w[1:] - high_1w[:-1]])
        down_move = np.concatenate([[np.nan], low_1w[:-1] - low_1w[1:]])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        period = 14
        alpha = 1.0 / period
        
        def wilders_smoothing(data, alpha):
            result = np.full_like(data, np.nan)
            for i in range(len(data)):
                if np.isnan(result[i-1]) if i > 0 else True:
                    result[i] = data[i]
                else:
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        tr_smoothed = wilders_smoothing(tr, alpha)
        plus_dm_smoothed = wilders_smoothing(plus_dm, alpha)
        minus_dm_smoothed = wilders_smoothing(minus_dm, alpha)
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (smoothed DX)
        adx = wilders_smoothing(dx, alpha)
        
        # Trend filter: ADX > 25 for strong trend, ADX < 20 for weak trend
        strong_trend = adx > 25
        weak_trend = adx < 20
    else:
        strong_trend = np.full(len(df_1w), False)
        weak_trend = np.full(len(df_1w), True)
    
    # Align 1w indicators to 12h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    weak_trend_aligned = align_htf_to_ltf(prices, df_1w, weak_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(strong_trend_aligned[i]) or 
            np.isnan(weak_trend_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND strong trend
            if (close[i] > donchian_upper[i] and 
                volume_filter[i] and 
                strong_trend_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND strong trend
            elif (close[i] < donchian_lower[i] and 
                  volume_filter[i] and 
                  strong_trend_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to midpoint OR weak trend
            if (close[i] < donchian_mid[i] or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to midpoint OR weak trend
            if (close[i] > donchian_mid[i] or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals