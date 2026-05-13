#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA200 trend filter and volume confirmation.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours. Discrete sizing 0.20.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Uses 4h HTF for trend direction to reduce noise and avoid overtrading.
# Exit via opposite Camarilla level break (R1/S1) for mean reversion in ranging markets.

name = "1h_Camarilla_R1S1_Breakout_4hEMA200_Volume_Session_v1"
timeframe = "1h"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead and TypeError
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12  # R1 = close + 1.1*(high-low)/12
    s1 = close_1d - 1.1 * camarilla_range / 12  # S1 = close - 1.1*(high-low)/12
    r3 = close_1d + 1.1 * camarilla_range       # R3 = close + 1.1*(high-low)
    s3 = close_1d - 1.1 * camarilla_range       # S3 = close - 1.1*(high-low)
    
    # Align 1d Camarilla levels to 1h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMA200 to 1h timeframe (wait for 4h bar to close)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade between 08:00-20:00 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 AND 4h EMA200 rising AND volume > 1.5x average
            if (close[i] > r1_aligned[i] and 
                ema_200_4h_aligned[i] > ema_200_4h_aligned[i-1] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 AND 4h EMA200 falling AND volume > 1.5x average
            elif (close[i] < s1_aligned[i] and 
                  ema_200_4h_aligned[i] < ema_200_4h_aligned[i-1] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 (mean reversion)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 (mean reversion)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals