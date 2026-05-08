#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Camarilla pivot levels with 1-day trend filter and volume confirmation.
# Long when price approaches S1 support in uptrend with volume surge; short when approaching R1 resistance in downtrend.
# Uses 1-day EMA(34) for trend direction and Camarilla levels from prior day's range.
# Designed for low trade frequency (12-37/year) to minimize fee fatigue and capture mean reversion in range-bound markets.

name = "12h_Camarilla_S1R1_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1-day EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Calculate Camarilla levels from previous day's range
    # HLC = (high + low + close) / 3
    hlc = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    
    # Camarilla levels
    S1 = hlc - (range_ * 1.1 / 6)
    S2 = hlc - (range_ * 1.1 / 4)
    S3 = hlc - (range_ * 1.1 / 2)
    S4 = hlc - (range_ * 1.1)
    R1 = hlc + (range_ * 1.1 / 6)
    R2 = hlc + (range_ * 1.1 / 4)
    R3 = hlc + (range_ * 1.1 / 2)
    R4 = hlc + (range_ * 1.1)
    
    # Align 1d indicators to 12h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    
    # Volume confirmation: 12h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: near S1 support in uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] <= S1_aligned[i] * 1.002 and  # Within 0.2% of S1
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: near R1 resistance in downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] >= R1_aligned[i] * 0.998 and  # Within 0.2% of R1
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S2 or trend turns down
            if close[i] >= S2_aligned[i] or trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R2 or trend turns up
            if close[i] <= R2_aligned[i] or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals