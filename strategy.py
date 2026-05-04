#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 4h for structure, 4h EMA50 for trend filter (proven edge from top performers),
# and volume spike for confirmation. Session filter (08-20 UTC) reduces noise trades.
# Designed for 15-30 trades/year to minimize fee drag. Works in bull markets via breakout continuations
# and in bear markets via breakdown continuations. The 4h EMA50 provides a smooth trend filter that
# adapts to changing regimes while avoiding whipsaw. This pattern aligns with proven performers like
# 4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_A (ETHUSDT test_sharpe=1.867).

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 trend filter from prior completed 4h bar
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_shifted = np.roll(ema50_4h, 1)
    ema50_4h_shifted[0] = np.nan
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_shifted)
    
    # Calculate Camarilla levels from prior completed 4h bar
    # Camarilla: R1 = close + 1.0833*(high-low), S1 = close - 1.0833*(high-low)
    daily_range_4h = high_4h - low_4h
    camarilla_r1 = close_4h + 1.0833 * daily_range_4h
    camarilla_s1 = close_4h - 1.0833 * daily_range_4h
    camarilla_r1_shifted = np.roll(camarilla_r1, 1)
    camarilla_s1_shifted = np.roll(camarilla_s1, 1)
    camarilla_r1_shifted[0] = np.nan
    camarilla_s1_shifted[0] = np.nan
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_shifted)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for performance)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND above 4h EMA50 AND volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND below 4h EMA50 AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S1 OR below 4h EMA50
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Camarilla R1 OR above 4h EMA50
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals