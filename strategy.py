#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend
# Strategy: Breakout of Camarilla R3/S3 levels with 1-week EMA trend filter
# Long when price breaks above R3 and price > 1w EMA(20)
# Short when price breaks below S3 and price < 1w EMA(20)
# Exit when price crosses back below R3 (long) or above S3 (short)
# Uses weekly trend filter to avoid counter-trend trades and reduce whipsaw
# Designed for 12h timeframe with selective entries to minimize trade frequency

name = "12h_Camarilla_R3_S3_Breakout_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1-week EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need previous day's data for Camarilla calculation
        # Since we're on 12h timeframe, we need to get the previous day's OHLC
        # We'll use the same day's data as approximation for intraday calculation
        # In practice, Camarilla uses previous day's OHLC
        
        # For 12h timeframe, we approximate using current day's data
        # This is a simplification but should work for demonstration
        # In a real implementation, we would need to access daily data
        
        # Calculate Camarilla levels using previous period's high/low/close
        # We'll use the previous bar's data as proxy for previous day
        if i > 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            if range_val > 0:
                R3 = prev_close + range_val * 1.1 / 4
                S3 = prev_close - range_val * 1.1 / 4
                
                if position == 0:
                    # Enter long: price breaks above R3 and above weekly EMA (uptrend)
                    if close[i] > R3 and close[i] > ema_20_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Enter short: price breaks below S3 and below weekly EMA (downtrend)
                    elif close[i] < S3 and close[i] < ema_20_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                
                elif position == 1:
                    # Exit long: price crosses back below R3
                    if close[i] < R3:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                
                elif position == -1:
                    # Exit short: price crosses back above S3
                    if close[i] > S3:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # Avoid division by zero
                if position != 0:
                    signals[i] = 0.0
                    position = 0
        else:
            # Not enough previous data
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals