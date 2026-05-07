#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: On 12h chart, enter long when price breaks above Camarilla R3 level with volume confirmation and 1d uptrend,
# enter short when price breaks below S3 level with volume confirmation and 1d downtrend.
# Uses 1d EMA34 for trend filter and volume spike confirmation to reduce false breakouts.
# Designed for low trade frequency (~15-30/year) to minimize fee drag and work in trending markets.
# Camarilla levels provide precise intraday support/resistance based on prior day's range.
# Works in both bull and bear markets by aligning with higher timeframe trend.
timeframe = "12h"
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    rang = high - low
    
    # Camarilla levels
    R3 = close + (rang * 1.1 / 4.0)  # Close + (range * 1.1/4)
    S3 = close - (rang * 1.1 / 4.0)  # Close - (range * 1.1/4)
    
    # 1-day EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if any critical value is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + 1d uptrend
            if (close[i] > R3[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + 1d downtrend
            elif (close[i] < S3[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below S3 (stoploss)
            if close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above R3 (stoploss)
            if close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals