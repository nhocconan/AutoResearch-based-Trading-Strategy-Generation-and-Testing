#!/usr/bin/env python3
# 4h_RangeBreakout_1dTrend_VolumeFilter
# Hypothesis: Range-bound markets (low volatility) followed by breakouts with volume and trend confirmation
# capture institutional moves. Uses 4h Bollinger Bands to detect range (low BB width), then breaks
# above upper BB or below lower BB with volume > 2x average and 1d EMA50 trend filter.
# Works in bull markets via long breakouts and bear via short breakdowns. Target: 25-40 trades/year.

name = "4h_RangeBreakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Bollinger Bands (20, 2) for range detection
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Range condition: BB width < 20th percentile (tight range)
    bb_width_series = pd.Series(bb_width)
    bb_width_p20 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    range_condition = bb_width < bb_width_p20
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for BB width percentile
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width[i]) or np.isnan(bb_width_p20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper[i]
        breakout_down = close[i] < bb_lower[i]
        
        if position == 0:
            # Enter long: range breakout up + volume + uptrend
            if range_condition[i] and breakout_up and vol_filter[i] and (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: range breakout down + volume + downtrend
            elif range_condition[i] and breakout_down and vol_filter[i] and (close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to middle band or trend reversal
            if close[i] < bb_mid[i] or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to middle band or trend reversal
            if close[i] > bb_mid[i] or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals