#!/usr/bin/env python3
name = "1D_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
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
    
    # Calculate Camarilla levels for each day using prior day's OHLC
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate current levels (no look-ahead)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        camarilla_R3[i] = prev_close + range_val * 1.1 / 4
        camarilla_S3[i] = prev_close - range_val * 1.1 / 4
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition
        vol_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R3 with weekly uptrend and volume
            if close[i] > camarilla_R3[i] and close[i] > ema34_1w_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with weekly downtrend and volume
            elif close[i] < camarilla_S3[i] and close[i] < ema34_1w_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Camarilla range (below R3) or trend reversal
            if close[i] < camarilla_R3[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Camarilla range (above S3) or trend reversal
            if close[i] > camarilla_S3[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals