#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d volume spike and 1w ADX trend filter
# Long when Williams %R < -80 (oversold) AND volume > 2.0x 20-period average AND 1w ADX > 25 (trending)
# Short when Williams %R > -20 (overbought) AND volume > 2.0x 20-period average AND 1w ADX > 25 (trending)
# Exit when Williams %R returns to -50 level OR 1w ADX drops below 20 (range market)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Williams %R identifies exhaustion points, volume spike confirms participation,
# 1w ADX ensures we only trade in trending environments to avoid false reversals in chop.
# Works in bull markets via longs on pullbacks and bear markets via shorts on rallies.

name = "6h_WilliamsR_EXTREME_1wADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.concatenate([[np.nan], high_1w[:-1]])) > 
                       (np.concatenate([[np.nan], low_1w[:-1]]) - low_1w),
                       np.maximum(high_1w - np.concatenate([[np.nan], high_1w[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[np.nan], low_1w[:-1]]) - low_1w) > 
                        (high_1w - np.concatenate([[np.nan], high_1w[:-1]])),
                        np.maximum(np.concatenate([[np.nan], low_1w[:-1]]) - low_1w, 0), 0)
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    tr_smoothed = wilder_smooth(tr, 14)
    dm_plus_smoothed = wilder_smooth(dm_plus, 14)
    dm_minus_smoothed = wilder_smooth(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    def ewm_wilder(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First ADX value is simple average of first 'period' DX values
            result[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            # Subsequent values: ADX = ((prev_ADX * (period-1)) + current DX) / period
            alpha = 1.0 / period
            for i in range(2*period-1, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(dx[i]):
                    result[i] = (result[i-1] * (1 - alpha)) + (alpha * dx[i])
        return result
    
    adx = ewm_wilder(dx, 14)
    
    # Trend filter: ADX > 25 indicates trending market
    adx_trending = adx > 25
    # Range filter: ADX < 20 indicates ranging market (for exit)
    adx_ranging = adx < 20
    
    # Align 1w indicators to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1w, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1w, adx_ranging.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_trending_aligned[i]) or 
            np.isnan(adx_ranging_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND volume spike AND 1w trending (ADX > 25)
            if (williams_r_aligned[i] < -80 and 
                volume_filter[i] and 
                adx_trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND volume spike AND 1w trending (ADX > 25)
            elif (williams_r_aligned[i] > -20 and 
                  volume_filter[i] and 
                  adx_trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR 1w ADX drops below 20 (ranging)
            if (williams_r_aligned[i] > -50 or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR 1w ADX drops below 20 (ranging)
            if (williams_r_aligned[i] < -50 or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals