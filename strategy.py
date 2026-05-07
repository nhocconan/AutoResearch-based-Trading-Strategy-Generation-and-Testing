#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, S3, S4
    # Using previous day's data
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    # Align to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough history for volatility filter
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extreme volatility (using 12h ATR-like measure)
        # Use price range as volatility proxy
        price_range = high[i] - low[i]
        # Calculate 20-period average range
        if i >= 20:
            range_sum = np.sum(high[i-20:i] - low[i-20:i])
            avg_range = range_sum / 20
            vol_filter = price_range < (2.0 * avg_range)
        else:
            vol_filter = True
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume + volatility filter
            if close[i] > R3_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume + volatility filter
            elif close[i] < S3_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite level
            if position == 1:  # Long position
                if close[i] < S3_aligned[i]:  # Exit if price breaks below S3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                if close[i] > R3_aligned[i]:  # Exit if price breaks above R3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals