#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME
# Hypothesis: 12h Camarilla R3/S3 breakouts with 1d trend filter (EMA34) and volume spike confirmation. 
# Camarilla levels provide institutional support/resistance. 
# Volume spike confirms institutional participation. 
# 1d EMA34 filter ensures trading with higher timeframe trend. 
# Designed for low-frequency, high-conviction trades (target: 20-40/year) to work in both bull and bear markets.
# Uses 1d HTF for trend filter as required.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME"
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
    
    # Calculate Camarilla levels for each bar using previous day's OHLC
    # For intraday, we use previous bar's values as proxy for previous day
    # In practice, Camarilla uses prior day's range, but for 12h timeframe we approximate
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN at start
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # R3 and S3 levels
    r3 = prev_close + range_val * 1.1 / 4
    s3 = prev_close - range_val * 1.1 / 4
    
    # Volume spike: current volume > 1.5 x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema1d_aligned[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above R3 with volume spike and uptrend (price > 1d EMA34)
            if close[i] > r3[i] and vol_spike[i] and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3 with volume spike and downtrend (price < 1d EMA34)
            elif close[i] < s3[i] and vol_spike[i] and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S3 (reversal signal) or trend breaks
            if close[i] < s3[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R3 (reversal signal) or trend breaks
            if close[i] > r3[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals