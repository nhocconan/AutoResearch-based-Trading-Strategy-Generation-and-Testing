#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND daily EMA34 > daily EMA89 AND volume > 2x 20-period average.
# Short when price breaks below Camarilla S3 AND daily EMA34 < daily EMA89 AND volume > 2x 20-period average.
# Exit when price crosses back inside the Camarilla H3/L3 range.
# Camarilla provides precise support/resistance, daily EMA filter ensures trend alignment, volume spike confirms institutional interest.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Camarilla_R3S3_Breakout_DailyTrend_Volume"
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
    
    # Daily data for Camarilla calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 89:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_daily_close = df_d['close'].shift(1).values
    prev_daily_high = df_d['high'].shift(1).values
    prev_daily_low = df_d['low'].shift(1).values
    daily_range = prev_daily_high - prev_daily_low
    
    # Camarilla levels: H3/L3 = C ± 1.1*(H-L)/2, R3/S3 = C ± 1.1*(H-L)
    camarilla_H3 = prev_daily_close + 1.1 * daily_range / 2
    camarilla_L3 = prev_daily_close - 1.1 * daily_range / 2
    camarilla_R3 = prev_daily_close + 1.1 * daily_range
    camarilla_S3 = prev_daily_close - 1.1 * daily_range
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_d, camarilla_H3)
    L3_aligned = align_htf_to_ltf(prices, df_d, camarilla_L3)
    R3_aligned = align_htf_to_ltf(prices, df_d, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_d, camarilla_S3)
    
    # Daily EMA trend filter
    daily_close = df_d['close'].values
    ema34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89 = pd.Series(daily_close).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_d, ema34)
    ema89_aligned = align_htf_to_ltf(prices, df_d, ema89)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(89, 20)  # Sufficient warmup for EMA89 and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(ema89_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, daily EMA34 > EMA89, volume filter
            long_cond = (close[i] > R3_aligned[i]) and (ema34_aligned[i] > ema89_aligned[i]) and volume_filter[i]
            # Short conditions