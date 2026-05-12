#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Use Camarilla R3/S3 breakout on daily timeframe with weekly EMA trend filter and volume spike.
# Long when price breaks above R3 with price > weekly EMA and volume > 2x MA.
# Short when price breaks below S3 with price < weekly EMA and volume > 2x MA.
# Exit when price reverses back into the Camarilla range (between S3 and R3).
# Designed to capture strong trending moves with confirmation, works in both bull and bear markets by filtering with weekly trend.
# Targets 15-25 trades/year to minimize fee drag (30-100 total over 4 years).

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Calculate Camarilla levels (based on previous day)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # Using previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    R3 = prev_close + 1.1 * (prev_high - prev_low)
    S3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with price > weekly EMA and volume > 2x MA
            if close[i] > R3[i] and close[i] > weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with price < weekly EMA and volume > 2x MA
            elif close[i] < S3[i] and close[i] < weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below S3 (into the Camarilla range)
            if close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back above R3 (into the Camarilla range)
            if close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals