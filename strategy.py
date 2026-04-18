#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_MeanReversion_Bounce
Hypothesis: Weekly pivot levels (R3/S3) act as strong support/resistance on daily timeframe.
Price tends to bounce from these levels with mean-reversion tendency. Works in both bull and bear
markets as price respects weekly structure. Uses 6h for entry timing with volume confirmation.
Target: 15-30 trades/year (60-120 total over 4 years) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_range = prev_weekly_high - prev_weekly_low
    
    # Weekly pivot point and support/resistance levels
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    r1 = 2 * weekly_pivot - prev_weekly_low
    s1 = 2 * weekly_pivot - prev_weekly_high
    r2 = weekly_pivot + prev_weekly_range
    s2 = weekly_pivot - prev_weekly_range
    r3 = weekly_pivot + 2 * prev_weekly_range
    s3 = weekly_pivot - 2 * prev_weekly_range
    
    # Align weekly levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # 6-day RSI for overbought/oversold confirmation (using close prices)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/6, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1/6, min_periods=6).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 6)  # Warmup for volume MA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(weekly_pivot_6h[i]) or np.isnan(volume_filter[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        pivot_val = weekly_pivot_6h[i]
        vol_ok = volume_filter[i]
        rsi_val = rsi_values[i]
        
        if position == 0:
            # Long: bounce from S3 with oversold RSI and volume
            if price <= s3_val * 1.005 and rsi_val < 30 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: bounce from R3 with overbought RSI and volume
            elif price >= r3_val * 0.995 and rsi_val > 70 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches weekly pivot or RSI overbought
            if price >= pivot_val * 0.995 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches weekly pivot or RSI oversold
            if price <= pivot_val * 1.005 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R3S3_MeanReversion_Bounce"
timeframe = "6h"
leverage = 1.0