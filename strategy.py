#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) for breakout entries, filtered by weekly EMA34 trend direction and volume surge after low volatility periods. Camarilla levels provide high-probability reversal/breakout points in ranging markets, while the weekly trend filter ensures alignment with higher timeframe momentum. Volume surge confirms breakout strength, and low volatility regime helps avoid false breakouts during chop. Designed for 1d timeframe to target 10-30 trades/year with discrete sizing (0.25) to minimize fee churn and improve generalization in both bull and bear markets.
"""

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)

    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily Camarilla levels (based on prior day's range)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use R3 and S3 for breakout entries
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    
    for i in range(1, n):
        # Use previous day's high, low, close
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        rang = prev_high - prev_low
        camarilla_r3[i] = prev_close + (rang * 1.1 / 4)
        camarilla_s3[i] = prev_close - (rang * 1.1 / 4)
    
    # For first bar, set to current close to avoid triggering
    camarilla_r3[0] = close[0]
    camarilla_s3[0] = close[0]
    
    # Volume indicators: 20-period average and volatility regime
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_std_20 = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = vol_std_20 < (vol_avg_50 * 0.5)  # volatility less than half of 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start from 20 to have enough data for volume indicators
        # Get aligned values for current daily bar
        ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)[i]
        vol_avg_val = vol_avg_20[i]
        low_vol = low_vol_regime[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_aligned) or np.isnan(vol_avg_val) or np.isnan(low_vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R3 + weekly uptrend + low vol regime + volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i-1] <= camarilla_r3[i-1] and  # breakout confirmation
                close[i] > ema34_aligned and 
                low_vol and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S3 + weekly downtrend + low vol regime + volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i-1] >= camarilla_s3[i-1] and  # breakdown confirmation
                  close[i] < ema34_aligned and 
                  low_vol and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below Camarilla S3 or weekly trend turns down
            if (close[i] < camarilla_s3[i] or close[i] < ema34_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above Camarilla R3 or weekly trend turns up
            if (close[i] > camarilla_r3[i] or close[i] > ema34_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals