#!/usr/bin/env python3
# 6h_1d_engulfing_reversal_v1
# Hypothesis: 6-hour reversals at daily support/resistance zones using candlestick engulfing patterns.
# Uses daily pivot points (classic: PP, R1, S1, R2, S2) as key levels.
# Bullish engulfing near S1/S2 with volume confirmation = long.
# Bearish engulfing near R1/R2 with volume confirmation = short.
# Works in bull markets (buy dips to support) and bear markets (sell rallies to resistance).
# Engulfing patterns signal momentum exhaustion/reversal at key levels.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_engulfing_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: PP = (H+L+C)/3
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    # Support and resistance levels
    s1 = 2 * daily_pivot - daily_high
    s2 = daily_pivot - (daily_high - daily_low)
    r1 = 2 * daily_pivot - daily_low
    r2 = daily_pivot + (daily_high - daily_low)
    
    # Align pivot levels to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                         (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                         (close[i] < open_price[i-1]) and (open_price[i] > close[i-1])
        
        if position == 1:  # Long position
            # Exit: bearish engulfing at resistance or price reaches R2
            if bearish_engulf and (close[i] >= r1_aligned[i] or close[i] >= r2_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish engulfing at support or price reaches S2
            if bullish_engulf and (close[i] <= s1_aligned[i] or close[i] <= s2_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish engulfing near support with volume confirmation
            near_support = (low[i] <= s1_aligned[i] * 1.005) or (low[i] <= s2_aligned[i] * 1.005)
            if bullish_engulf and near_support and volume[i] > vol_ma_24[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish engulfing near resistance with volume confirmation
            near_resistance = (high[i] >= r1_aligned[i] * 0.995) or (high[i] >= r2_aligned[i] * 0.995)
            if bearish_engulf and near_resistance and volume[i] > vol_ma_24[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals