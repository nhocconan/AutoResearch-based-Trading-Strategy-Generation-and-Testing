#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d ADX Trend Filter and Volume Spike Confirmation.
- Primary timeframe: 6h for lower trade frequency and better signal quality vs lower TFs.
- HTF: 1d ADX(14) for trend strength (ADX > 25 = trending market).
- Williams %R(14) from 6h: Extreme oversold (< -80) for long, extreme overbought (> -20) for short.
- Volume: Current 6h volume > 1.8 * 20-period volume MA to capture institutional participation.
- Entry: Long when Williams %R < -80 AND 1d ADX > 25 AND volume spike.
         Short when Williams %R > -20 AND 1d ADX > 25 AND volume spike.
- Exit: Williams %R reverts to -50 (mean reversion) OR loss of volume confirmation OR ADX < 20 (trend weakens).
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe.
This strategy captures mean reversion in strong trends (ADX > 25) where Williams %R extremes
indicate exhaustion, with volume confirmation ensuring institutional participation. Works in
both bull and bear markets by only trading in the direction of the strong trend (ADX filter),
avoiding choppy markets where mean reversion fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high - df_1d_low)
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low),
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)),
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 6h Williams %R(14)
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0,
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Calculate 20-period 6h volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)  # Use 1d vol MA as reference
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period volume MA
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough bars for ADX and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals with volume spike and strong trend (ADX > 25)
            if volume_spike[i] and adx_aligned[i] > 25:
                # Long entry: Williams %R extremely oversold (< -80)
                if wr[i] < -80:
                    signals[i] = 0.25
                    position = 1
                # Short entry: Williams %R extremely overbought (> -20)
                elif wr[i] > -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R reverts to -50 OR loss of volume confirmation OR trend weakens (ADX < 20)
            if wr[i] > -50 or not volume_spike[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reverts to -50 OR loss of volume confirmation OR trend weakens (ADX < 20)
            if wr[i] < -50 or not volume_spike[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0