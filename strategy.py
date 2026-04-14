#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Trend_v1
Hypothesis: On daily timeframe, use weekly Donchian channel breakouts with volume confirmation and ADX trend filter.
Go long when price breaks above weekly Donchian high (20-period) with volume surge in strong trend (ADX > 25).
Go short when price breaks below weekly Donchian low with volume surge in strong trend.
Exit when price crosses the weekly Donchian midpoint or ADX weakens below 20.
Designed to capture major trends in both bull and bear markets while avoiding false breakouts in low-volume, low-volatility conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = np.full_like(high_weekly, np.nan)
    donchian_low = np.full_like(low_weekly, np.nan)
    donchian_mid = np.full_like(close_weekly, np.nan)
    
    for i in range(19, len(high_weekly)):
        if np.isnan(high_weekly[i-19:i+1]).any() or np.isnan(low_weekly[i-19:i+1]).any():
            continue
        donchian_high[i] = np.max(high_weekly[i-19:i+1])
        donchian_low[i] = np.min(low_weekly[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Calculate ADX on weekly data (14-period)
    if len(high_weekly) < 14:
        return np.zeros(n)
    
    plus_dm = np.zeros_like(high_weekly)
    minus_dm = np.zeros_like(high_weekly)
    tr = np.zeros_like(high_weekly)
    
    for i in range(1, len(high_weekly)):
        if np.isnan(high_weekly[i]) or np.isnan(low_weekly[i]) or np.isnan(high_weekly[i-1]) or np.isnan(low_weekly[i-1]):
            continue
        high_diff = high_weekly[i] - high_weekly[i-1]
        low_diff = low_weekly[i-1] - low_weekly[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_weekly[i] - low_weekly[i], 
                   abs(high_weekly[i] - high_weekly[i-1]), 
                   abs(low_weekly[i] - low_weekly[i-1]))
    
    # Wilder's smoothing for TR, +DM, -DM
    atr = np.zeros_like(high_weekly)
    plus_di = np.zeros_like(high_weekly)
    minus_di = np.zeros_like(high_weekly)
    dx = np.zeros_like(high_weekly)
    adx = np.full_like(high_weekly, np.nan)
    
    if len(high_weekly) >= 14:
        # Initial values (first 14 periods)
        atr[13] = np.nansum(tr[1:14])
        plus_dm_sum = np.nansum(plus_dm[1:14])
        minus_dm_sum = np.nansum(minus_dm[1:14])
        
        for i in range(14, len(high_weekly)):
            if np.isnan(tr[i]) or np.isnan(plus_dm[i]) or np.isnan(minus_dm[i]):
                atr[i] = atr[i-1]
                plus_dm_sum = plus_dm_sum
                minus_dm_sum = minus_dm_sum
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                plus_dm_sum = (plus_dm_sum * 13 + plus_dm[i]) / 14
                minus_dm_sum = (minus_dm_sum * 13 + minus_dm[i]) / 14
            
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_sum / atr[i]
                minus_di[i] = 100 * minus_dm_sum / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Calculate ADX as smoothed DX
        if len(high_weekly) >= 27:
            adx[26] = np.nanmean(dx[14:27])
            for i in range(27, len(high_weekly)):
                if np.isnan(dx[i]):
                    adx[i] = adx[i-1]
                else:
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.30  # 30% position size
    
    for i in range(20, n):  # Start after enough data for alignment
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Look for long entries: price breaks above weekly Donchian high with volume surge in strong trend
            if (close[i] > donchian_high_aligned[i] and
                volume_ratio > 2.0 and
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Look for short entries: price breaks below weekly Donchian low with volume surge in strong trend
            elif (close[i] < donchian_low_aligned[i] and
                  volume_ratio > 2.0 and
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses weekly Donchian midpoint or ADX weakens
            if (close[i] < donchian_mid_aligned[i] or
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses weekly Donchian midpoint or ADX weakens
            if (close[i] > donchian_mid_aligned[i] or
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Weekly_Donchian_Trend_v1"
timeframe = "1d"
leverage = 1.0