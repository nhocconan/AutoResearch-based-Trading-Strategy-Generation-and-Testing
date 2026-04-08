#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: Camarilla pivot levels from 1d timeframe act as strong support/resistance on 12h.
# Long when price touches S3 (strong support) with volume spike in bullish regime (ADX < 20).
# Short when price touches R3 (strong resistance) with volume spike in bullish regime (ADX < 20).
# Uses ADX < 20 to filter for ranging markets where pivot reversals work best.
# Volume confirmation: current volume > 1.5 * average volume (20 periods).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate average volume for confirmation (20 periods)
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Camarilla levels:
    # S3 = Close - (Range * 1.1000 / 4)
    # S2 = Close - (Range * 1.1000 / 6)
    # S1 = Close - (Range * 1.1000 / 12)
    # R1 = Close + (Range * 1.1000 / 12)
    # R2 = Close + (Range * 1.1000 / 6)
    # R3 = Close + (Range * 1.1000 / 4)
    s3_1d = close_1d - (range_1d * 1.1000 / 4.0)
    r3_1d = close_1d + (range_1d * 1.1000 / 4.0)
    
    # Align 1d Camarilla levels to 12h timeframe
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Calculate ADX for regime filter (14 periods)
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(adx[i]) or 
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * average volume
        volume_spike = volume[i] > 1.5 * avg_volume[i]
        
        # Regime filter: ADX < 20 (ranging market)
        ranging_market = adx[i] < 20
        
        if position == 1:  # Long position
            # Exit: price moves above S1 (profit taking) or opposite signal
            # S1 = Close - (Range * 1.1000 / 12)
            s1_1d = close_1d - (range_1d * 1.1000 / 12.0)
            s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
            
            if not np.isnan(s1_1d_aligned[i]) and close[i] > s1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif (low[i] <= s3_1d_aligned[i] * 1.005 and  # Still near S3
                  volume_spike and ranging_market):
                signals[i] = 0.25  # Hold long
            else:
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit: price moves below R1 (profit taking) or opposite signal
            # R1 = Close + (Range * 1.1000 / 12)
            r1_1d = close_1d + (range_1d * 1.1000 / 12.0)
            r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
            
            if not np.isnan(r1_1d_aligned[i]) and close[i] < r1_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif (high[i] >= r3_1d_aligned[i] * 0.995 and  # Still near R3
                  volume_spike and ranging_market):
                signals[i] = -0.25  # Hold short
            else:
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Long entry: price touches S3 with volume spike in ranging market
            if (low[i] <= s3_1d_aligned[i] * 1.005 and  # Touched or slightly below S3
                volume_spike and ranging_market):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R3 with volume spike in ranging market
            elif (high[i] >= r3_1d_aligned[i] * 0.995 and  # Touched or slightly above R3
                  volume_spike and ranging_market):
                position = -1
                signals[i] = -0.25
    
    return signals