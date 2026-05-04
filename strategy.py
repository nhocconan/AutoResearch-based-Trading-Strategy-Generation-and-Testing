#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from prior 4h bar (R3/S3) for structure, 12h EMA50 for trend filter
# Volume confirmation (>1.5x 20 EMA) ensures breakout has participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in both bull and bear: EMA50 ensures we only trade with the trend, Camarilla provides objective breakout levels.

name = "4h_Camarilla_R3S3_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Camarilla levels from prior 4h bar (use prior completed bar only)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C, H, L are from prior completed bar
    prior_close = pd.Series(close).shift(1).values
    prior_high = pd.Series(high).shift(1).values
    prior_low = pd.Series(low).shift(1).values
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4.0
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4.0
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price above 12h EMA50 + volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema_50_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price below 12h EMA50 + volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema_50_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point OR price crosses below 12h EMA50
            camarilla_pivot = (prior_high[i] + prior_low[i] + prior_close[i]) / 3.0
            if not np.isnan(camarilla_pivot) and (close[i] < camarilla_pivot or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point OR price crosses above 12h EMA50
            camarilla_pivot = (prior_high[i] + prior_low[i] + prior_close[i]) / 3.0
            if not np.isnan(camarilla_pivot) and (close[i] > camarilla_pivot or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals