#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dATR_VolumeSpike_TrendFilter
Hypothesis: Trade Camarilla R3/S3 breakouts with 1d ATR-based trend filter (more adaptive than fixed EMA) and volume spike confirmation.
ATR trend filter adapts to volatility regimes, reducing whipsaws in both bull and bear markets. R3/S3 are stronger levels reducing false breakouts.
Only trade in direction of 1d trend (using ATR-based adaptive trend) to avoid counter-trend whipsaws. Discrete sizing 0.25 to manage risk and minimize fee churn.
Target: 25-45 trades/year to stay within fee drag limits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14) for volatility adaptive trend filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(np.roll(close_1d, 1) - low_1d))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate adaptive trend: price > close + 0.5*ATR = bullish, price < close - 0.5*ATR = bearish
    trend_bullish = close_1d > (close_1d + 0.5 * atr_14_1d)
    trend_bearish = close_1d < (close_1d - 0.5 * atr_14_1d)
    # Actually, we need to compare current price to previous close +/- ATR
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    trend_bullish = close_1d > (prev_close_1d + 0.5 * atr_14_1d)
    trend_bearish = close_1d < (prev_close_1d - 0.5 * atr_14_1d)
    
    # Align trend filters to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r3 = prev_day_close + 1.1 * camarilla_range / 4  # R3 level
    s3 = prev_day_close - 1.1 * camarilla_range / 4  # S3 level
    h3 = prev_day_close + 1.1 * camarilla_range / 6  # H3 level
    l3 = prev_day_close - 1.1 * camarilla_range / 6  # L3 level
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0x 20-period average (stricter filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily ATR (14) and volume MA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND daily trend bullish AND volume spike
            long_setup = (close[i] > r3_aligned[i]) and \
                         (trend_bullish_aligned[i] > 0.5) and \
                         volume_spike[i]
            # Short: price breaks below S3 AND daily trend bearish AND volume spike
            short_setup = (close[i] < s3_aligned[i]) and \
                          (trend_bearish_aligned[i] > 0.5) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bearish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bullish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dATR_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0