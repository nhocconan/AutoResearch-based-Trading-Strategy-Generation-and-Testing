#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 reversal with 1d trend filter and volume confirmation.
# Fade at R3/S3 when 1d trend is opposite (mean reversion in range),
# Breakout continuation at R4/S4 when 1d trend aligns (trend follow).
# Uses 1d EMA(34) for trend and 60-period volume spike for confirmation.
# Target: 60-100 total trades over 4 years (15-25/year) to minimize fee drag.

name = "6h_Camarilla_R3S3_Reversal_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    range_ = prev_high - prev_low
    R3 = prev_close + (range_ * 1.1 / 2)
    S3 = prev_close - (range_ * 1.1 / 2)
    R4 = prev_close + (range_ * 1.1)
    S4 = prev_close - (range_ * 1.1)
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 60-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=60, adjust=False, min_periods=60).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 1d indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry conditions
            # Fade at S3 in downtrend (mean reversion)
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                close[i] <= S3_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Breakout above R4 in uptrend (trend follow)
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  close[i] >= R4_aligned[i] and
                  vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry conditions
            # Fade at R3 in uptrend (mean reversion)
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  close[i] >= R3_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            # Breakdown below S4 in downtrend (trend follow)
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] <= S4_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or stop
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                close[i] >= R3_aligned[i]):  # Hit R3 or above
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  close[i] <= S4_aligned[i]):  # Break below S4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or stop
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] <= S3_aligned[i]):  # Hit S3 or below
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] >= R4_aligned[i]):  # Break above R4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals