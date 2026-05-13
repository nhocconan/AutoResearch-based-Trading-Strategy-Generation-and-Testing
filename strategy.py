#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Daily range for Camarilla levels (use previous day's range)
    prev_day_high = np.roll(high, 1)
    prev_day_low = np.roll(low, 1)
    prev_day_close = np.roll(close, 1)
    prev_day_high[0] = high[0]
    prev_day_low[0] = low[0]
    prev_day_close[0] = close[0]
    
    # Calculate Camarilla levels for each day (using previous day's data)
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    for i in range(n):
        range_val = prev_day_high[i] - prev_day_low[i]
        camarilla_r1[i] = prev_day_close[i] + range_val * 1.1 / 12
        camarilla_s1[i] = prev_day_close[i] - range_val * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        price_above_r1 = close[i] > camarilla_r1[i]
        price_below_s1 = close[i] < camarilla_s1[i]
        
        if position == 0:
            # LONG: price breaks above R1 + 1d uptrend + volume confirmation
            if price_above_r1 and close[i] > ema34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + 1d downtrend + volume confirmation
            elif price_below_s1 and close[i] < ema34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or trend breaks
            if price_below_s1 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or trend breaks
            if price_above_r1 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals