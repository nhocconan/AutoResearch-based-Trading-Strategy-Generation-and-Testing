#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 12h ADX Trend Filter and Volume Spike Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ADX for trend strength (ADX > 25 = trending market).
- Williams %R(14) for extreme readings: Long when %R crosses above -80 from below, Short when %R crosses below -20 from above.
- Volume confirmation: Current volume > 1.5 * 20-period volume MA.
- Exit: Reverse signal or Williams %R returns to opposite extreme zone (%R > -20 for longs, %R < -80 for shorts).
- Signal size: 0.25 discrete to balance return and drawdown.
- Designed to work in both bull and bear markets by using ADX to filter for trending conditions and Williams %R for mean-reversion within trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if period < len(data):
            result[period-1] = np.nanmean(data[:period])
        # Rest: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smoothing(dx, 14)
    
    # Get 6h data for Williams %R
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need enough bars for Williams %R and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation and ADX filter
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            strong_trend = adx_aligned[i] > 25
            
            # Long: Williams %R crosses above -80 from below AND strong trend AND volume confirmed
            if (curr_williams_r > -80 and 
                i > start_idx and williams_r[i-1] <= -80 and 
                strong_trend and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND strong trend AND volume confirmed
            elif (curr_williams_r < -20 and 
                  i > start_idx and williams_r[i-1] >= -20 and 
                  strong_trend and vol_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R returns above -20 (overbought) or reverse signal
            if curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R returns below -80 (oversold) or reverse signal
            if curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0