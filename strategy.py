#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla R3/S3 levels act as strong support/resistance on 1d timeframe.
# In 1d uptrend (price > EMA34), go long when price breaks above R3 with volume confirmation.
# In 1d downtrend (price < EMA34), go short when price breaks below S3 with volume confirmation.
# Uses volume spike (1.5x 20-period average) to confirm breakouts.
# Trend filter prevents counter-trend trades, improving performance in both bull and bear markets.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume spike detection: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + price breaks above R3 + volume spike
            if (close[i] > ema34_aligned[i] and 
                close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + price breaks below S3 + volume spike
            elif (close[i] < ema34_aligned[i] and 
                  close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price drops below S3
            if (close[i] <= ema34_aligned[i] or 
                close[i] < camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price rises above R3
            if (close[i] >= ema34_aligned[i] or 
                close[i] > camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals