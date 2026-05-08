#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla levels from previous day
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Calculate Camarilla R3, R4, S3, S4 levels
    # R4 = Close + ((High - Low) * 1.500)
    # R3 = Close + ((High - Low) * 1.250)
    # S3 = Close - ((High - Low) * 1.250)
    # S4 = Close - ((High - Low) * 1.500)
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.250
    camarilla_r4 = close_prev + (high_prev - low_prev) * 1.500
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.250
    camarilla_s4 = close_prev - (high_prev - low_prev) * 1.500
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day for previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(close_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_prev > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike detection: current volume > 2.0 * 24-period average (24*12h = 12 days)
    volume = prices['volume'].values
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma24 * 2.0)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R4 with volume spike and daily uptrend
            long_cond = (high[i] > camarilla_r4_aligned[i] and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below S4 with volume spike and daily downtrend
            short_cond = (low[i] < camarilla_s4_aligned[i] and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (mean reversion to opposite side)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 (mean reversion to opposite side)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R4/S4 breakout with volume confirmation and daily trend filter on 12h timeframe.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite Camarilla levels).
# Uses proper Camarilla calculation: R3=Close+(H-L)*1.25, R4=Close+(H-L)*1.5, S3=Close-(H-L)*1.25, S4=Close-(H-L)*1.5.
# Volume spike requires 2x 24-period average (12 days of 12h data) to ensure significance.
# Daily EMA34 trend filter ensures alignment with longer-term trend, reducing counter-trend trades.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.