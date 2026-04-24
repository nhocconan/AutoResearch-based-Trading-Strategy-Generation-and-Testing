#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 4h for lower trade frequency and better signal quality.
- HTF: 1d ADX(14) > 25 for strong trend (bullish if +DI > -DI, bearish if -DI > +DI).
- Volume: Current 4h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Williams %R: Long when %R crosses above -80 from below (oversold reversal).
                Short when %R crosses below -20 from above (overbought reversal).
- Exit: Reverse signal or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe.
This strategy captures mean-reversion moves within strong trends, filtered by daily trend
direction to avoid counter-trend trades. Williams %R identifies exhaustion points, while
volume confirmation ensures institutional participation. Works in both bull and bear markets
by only taking trades in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX components
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high[1:] - df_1d_low[1:])
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((df_1d_high[1:] - df_1d_high[:-1]) > (df_1d_low[:-1] - df_1d_low[1:]),
                       np.maximum(df_1d_high[1:] - df_1d_high[:-1], 0), 0)
    dm_minus = np.where((df_1d_low[:-1] - df_1d_low[1:]) > (df_1d_high[1:] - df_1d_high[:-1]),
                        np.maximum(df_1d_low[:-1] - df_1d_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    tr_smoothed = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=alpha, adjust=False).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=alpha, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Trend direction: bullish if +DI > -DI, bearish if -DI > +DI
    trend_bullish = di_plus > di_minus
    trend_bearish = di_minus > di_plus
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R on 1d
    period_wr = 14
    highest_high = pd.Series(df_1d_high).rolling(window=period_wr, min_periods=period_wr).max().values
    lowest_low = pd.Series(df_1d_low).rolling(window=period_wr, min_periods=period_wr).min().values
    williams_r = -100 * (highest_high - df_1d_close) / (highest_high - lowest_low)
    
    # Align HTF indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 14)  # Need enough bars for ADX, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in strong trends (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for reversal signals with volume spike and strong trend
            if volume_spike[i] and strong_trend:
                # Bullish reversal: Williams %R crosses above -80 from below
                if i > 0 and williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80 and trend_bullish_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above
                elif i > 0 and williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20 and trend_bearish_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: reverse signal or loss of volume confirmation or weak trend
            if (i > 0 and williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20) or \
               not volume_spike[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or loss of volume confirmation or weak trend
            if (i > 0 and williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80) or \
               not volume_spike[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dADX_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0