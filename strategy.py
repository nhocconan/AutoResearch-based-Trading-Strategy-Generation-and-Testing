#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h timeframe with 1d ATR-based trend filter and volume confirmation.
The 1d ATR trend filter adapts to volatility regimes, providing better trend detection than fixed EMAs in both bull and bear markets.
R1/S1 levels offer tighter breakouts than R3/S3, increasing trade frequency while volume spike and ATR filter reduce false signals.
Designed for 12h to target 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25 to manage fee drag.
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
    
    # Get daily data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14) for trend filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Daily trend: close > close - 1.0 * ATR(14) = bullish, close < close + 1.0 * ATR(14) = bearish
    # Using 1-period lag to avoid look-ahead
    trend_bullish_1d = close_1d > (np.roll(close_1d, 1) + 1.0 * np.roll(atr_14_1d, 1))
    trend_bearish_1d = close_1d < (np.roll(close_1d, 1) - 1.0 * np.roll(atr_14_1d, 1))
    # Handle first value
    trend_bullish_1d[0] = False
    trend_bearish_1d[0] = False
    
    # Align daily trend to 12h timeframe
    trend_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r1 = prev_day_close + 1.1 * camarilla_range / 12  # R1 level
    s1 = prev_day_close - 1.1 * camarilla_range / 12  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily ATR (14) and volume MA (20)
    start_idx = max(20, 14) + 1  # +1 for shift(1) in Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(trend_bullish_1d_aligned[i]) or
            np.isnan(trend_bearish_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND daily trend bullish AND volume spike
            long_setup = (close[i] > r1_aligned[i]) and \
                         trend_bullish_1d_aligned[i] and \
                         volume_spike[i]
            # Short: price breaks below S1 AND daily trend bearish AND volume spike
            short_setup = (close[i] < s1_aligned[i]) and \
                          trend_bearish_1d_aligned[i] and \
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
            # Calculate H3/L3 for exit condition
            h3 = prev_day_close + 1.1 * camarilla_range / 6
            l3 = prev_day_close - 1.1 * camarilla_range / 6
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               trend_bearish_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bullish
            h3 = prev_day_close + 1.1 * camarilla_range / 6
            l3 = prev_day_close - 1.1 * camarilla_range / 6
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               trend_bullish_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0