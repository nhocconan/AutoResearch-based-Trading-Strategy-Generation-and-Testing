#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d RSI filter and volume spike.
# Long when price touches S3 level AND 1d RSI < 30 (oversold) AND volume > 2x 20-period average.
# Short when price touches R3 level AND 1d RSI > 70 (overbought) AND volume > 2x 20-period average.
# Exit when price crosses back to the pivot point (PP).
# Uses mean reversion at extreme pivot levels with RSI filter to avoid false signals in strong trends.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drag.

name = "12h_Camarilla_RSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h pivot point (PP) from previous bar
    pp = (high[:-1] + low[:-1] + close[:-1]) / 3
    pp = np.concatenate([[np.nan], pp])  # align with current bar
    
    # Calculate Camarilla levels (based on previous bar's range)
    range_val = (high[:-1] - low[:-1])
    r3 = pp + 1.1 * (range_val / 2)
    s3 = pp - 1.1 * (range_val / 2)
    
    # Align levels to current bar (use previous bar's levels for current bar)
    r3 = np.concatenate([[np.nan], r3[:-1]])
    s3 = np.concatenate([[np.nan], s3[:-1]])
    
    # 12h volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1d data
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rs)] = 50  # neutral when no loss
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: touch S3, RSI < 30, volume spike
            long_cond = (low[i] <= s3[i]) and (rsi_aligned[i] < 30) and volume_filter[i]
            # Short conditions: touch R3, RSI > 70, volume spike
            short_cond = (high[i] >= r3[i]) and (rsi_aligned[i] > 70) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back to pivot point
            if close[i] >= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back to pivot point
            if close[i] <= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals