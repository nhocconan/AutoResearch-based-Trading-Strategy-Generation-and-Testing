#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1-2-3 pattern with 1d RSI filter and volume confirmation.
# Long when price breaks above point 2 of 1-2-3 pattern AND 1d RSI > 50 AND volume > 1.5x 20-period average.
# Short when price breaks below point 1 of 1-2-3 pattern AND 1d RSI < 50 AND volume > 1.5x 20-period average.
# Exit when price crosses point 3 for long or point 2 for short.
# Uses 1-2-3 pattern for trend reversal with RSI filter to avoid counter-trend entries.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drift.

name = "4h_123Pattern_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h 1-2-3 pattern: point 1 (recent low), point 2 (recent high), point 3 (recent low after point 2)
    # We'll use 10-period lookback for simplicity
    lookback = 10
    point1 = np.full(n, np.nan)
    point2 = np.full(n, np.nan)
    point3 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Point 1: lowest low in lookback period
        point1[i] = np.min(low[i-lookback:i])
        # Point 2: highest high after point 1 formation (simplified: highest high in lookback)
        point2[i] = np.max(high[i-lookback:i])
        # Point 3: lowest low after point 2 (simplified: lowest low in lookback)
        point3[i] = np.min(low[i-lookback:i])
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1d data
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Align 1d RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(point1[i]) or np.isnan(point2[i]) or np.isnan(point3[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above point 2, RSI > 50, volume spike
            long_cond = (close[i] > point2[i]) and (rsi_aligned[i] > 50) and volume_filter[i]
            # Short conditions: break below point 1, RSI < 50, volume spike
            short_cond = (close[i] < point1[i]) and (rsi_aligned[i] < 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below point 3
            if close[i] < point3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above point 2
            if close[i] > point2[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals