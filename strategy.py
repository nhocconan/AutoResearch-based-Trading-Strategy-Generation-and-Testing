#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot-based mean reversion with volume confirmation
# - Uses weekly Camarilla pivot levels (R3/S3) for mean reversion entries
# - Uses daily trend filter (price above/below 20 EMA) to align with higher timeframe
# - Uses 6h volume spike for entry confirmation
# - Enters long when price touches S3 with bullish daily trend and volume
# - Enters short when price touches R3 with bearish daily trend and volume
# - Exits when price reaches the weekly pivot (midpoint) or opposite S3/R3 level
# - Designed to capture mean reversion moves in ranging markets while respecting trend
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "6h_WeeklyCamarilla_R3S3_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3
    # Weekly range
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3_1w = pp_1w + (range_1w * 1.1 / 4)  # R3 = PP + 1.1 * range / 4
    s3_1w = pp_1w - (range_1w * 1.1 / 4)  # S3 = PP - 1.1 * range / 4
    r4_1w = pp_1w + (range_1w * 1.1 / 2)  # R4 = PP + 1.1 * range / 2
    s4_1w = pp_1w - (range_1w * 1.1 / 2)  # S4 = PP - 1.1 * range / 2
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_1w_6h = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_6h = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_6h = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_6h = align_htf_to_ltf(prices, df_1w, s4_1w)
    pp_1w_6h = align_htf_to_ltf(prices, df_1w, pp_1w)  # Pivot for exit
    
    # Daily trend filter: price above/below 20 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_6h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily trend: 1 = bullish (price > EMA20), -1 = bearish (price < EMA20)
    daily_trend = np.where(close_1d > ema_20_1d, 1, -1)
    daily_trend_6h = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1w_6h[i]) or np.isnan(s3_1w_6h[i]) or 
            np.isnan(pp_1w_6h[i]) or np.isnan(daily_trend_6h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S3 with bullish daily trend and volume spike
            if (low[i] <= s3_1w_6h[i] and 
                daily_trend_6h[i] == 1 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches R3 with bearish daily trend and volume spike
            elif (high[i] >= r3_1w_6h[i] and 
                  daily_trend_6h[i] == -1 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches pivot (PP) or breaks below S4 (stop)
            if close[i] >= pp_1w_6h[i] or low[i] <= s4_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches pivot (PP) or breaks above R4 (stop)
            if close[i] <= pp_1w_6h[i] or high[i] >= r4_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals