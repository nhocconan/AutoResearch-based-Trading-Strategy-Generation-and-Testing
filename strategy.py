#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Reversal_With_1d_Trend_Filter
# Hypothesis: Camarilla pivot levels (S3/S4 for long, R3/R4 for short) on daily timeframe act as strong support/resistance.
# Enter on reversal from these levels with 1d trend alignment and volume confirmation.
# Designed for low-frequency, high-conviction trades (target: 15-35 trades/year) with clear risk/reward.

name = "12h_Camarilla_Pivot_Reversal_With_1d_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's range)
    # Using previous day's high, low, close to calculate today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero and handle first bar
    valid_prev = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
    
    # Camarilla formulas
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    range_hl = prev_high - prev_low
    R4 = prev_close + (range_hl * 1.1 / 2)
    R3 = prev_close + (range_hl * 1.1 / 4)
    S3 = prev_close - (range_hl * 1.1 / 4)
    S4 = prev_close - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 12h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need at least 1 day of previous data + EMA + volume MA
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price at or below S3/S4 + daily uptrend + volume spike
            # Using S3 as primary level, S4 as stronger confirmation
            if ((close[i] <= S3_12h[i] * 1.002) or (close[i] <= S4_12h[i] * 1.002)) and \
               uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price at or above R3/R4 + daily downtrend + volume spike
            elif ((close[i] >= R3_12h[i] * 0.998) or (close[i] >= R4_12h[i] * 0.998)) and \
                 downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above R3 or daily trend turns down
            if close[i] >= R3_12h[i] * 0.998 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below S3 or daily trend turns up
            if close[i] <= S3_12h[i] * 1.002 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals