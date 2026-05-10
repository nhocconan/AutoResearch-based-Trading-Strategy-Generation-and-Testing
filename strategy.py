#!/usr/bin/env python3
# 1H_4H_1D_Camarilla_R1S1_Breakout_Trend_Filter
# Hypothesis: Combine 1d trend filter with 4h Camarilla pivot breakout for direction, using 1h for precise entry.
# Long when: 1d close > EMA50 (uptrend) AND price breaks above 4h Camarilla R1 with volume confirmation.
# Short when: 1d close < EMA50 (downtrend) AND price breaks below 4h Camarilla S1 with volume confirmation.
# Uses volume spike (>1.5x 20-period average) to avoid false breakouts.
# Session filter (08-20 UTC) to reduce noise. Fixed size 0.20.
# Target: 20-40 trades/year per symbol.

name = "1H_4H_1D_Camarilla_R1S1_Breakout_Trend_Filter"
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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    # Since we don't have daily data in 4h, we approximate using 4h data
    # Camarilla uses previous day's range, but we'll use 4-period lookback as proxy
    # For better accuracy, we could use daily data, but 4h is acceptable proxy
    P = (high_4h[-4] + low_4h[-4] + close_4h[-4]) / 3  # Simplified: use last available 4h bar's OHLC
    R1 = P + (high_4h[-4] - low_4h[-4]) * 1.1 / 12
    S1 = P - (high_4h[-4] - low_4h[-4]) * 1.1 / 12
    
    # For simplicity and to avoid look-ahead, we'll use a rolling window approach
    # Calculate Camarilla levels for each 4h bar using prior day's data
    window = 6  # Approximately 1 day of 4h data (6*4h = 24h)
    if len(high_4h) < window:
        return np.zeros(n)
    
    # Calculate typical price for window
    typical_price = (high_4h + low_4h + close_4h) / 3
    P_roll = pd.Series(typical_price).rolling(window=window, min_periods=window).mean().values
    range_roll = (pd.Series(high_4h).rolling(window=window, min_periods=window).max() - 
                  pd.Series(low_4h).rolling(window=window, min_periods=window).min()).values
    
    R1_4h = P_roll + range_roll * 1.1 / 12
    S1_4h = P_roll - range_roll * 1.1 / 12
    
    # Align Camarilla levels to 1h
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if not in session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: 1d uptrend + price breaks above 4h R1 + volume spike
            if (close[i] > ema50_1d_aligned[i] and    # 1d uptrend
                high[i] > R1_4h_aligned[i] and       # Break above R1
                volume_spike[i]):                    # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Short condition: 1d downtrend + price breaks below 4h S1 + volume spike
            elif (close[i] < ema50_1d_aligned[i] and # 1d downtrend
                  low[i] < S1_4h_aligned[i] and      # Break below S1
                  volume_spike[i]):                  # Volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down OR price breaks below S1
            if (close[i] < ema50_1d_aligned[i] or    # 1d downtrend
                low[i] < S1_4h_aligned[i]):          # Break below S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 1d trend turns up OR price breaks above R1
            if (close[i] > ema50_1d_aligned[i] or    # 1d uptrend
                high[i] > R1_4h_aligned[i]):         # Break above R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals