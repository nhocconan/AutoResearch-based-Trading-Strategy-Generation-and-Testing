#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with daily trend filter and volume confirmation
# Enter long when: price breaks above upper BB(20,2) AND price > 1d EMA(50) AND volume > 1.5x avg
# Enter short when: price breaks below lower BB(20,2) AND price < 1d EMA(50) AND volume > 1.5x avg
# Exit when price returns to middle BB(20,2) or opposite breakout occurs
# Uses Bollinger squeeze (low volatility) to anticipate breakouts, targeting 100-200 trades over 4 years

name = "4h_bb_squeeze_breakout_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) on 4h
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for BB to stabilize
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < middle BB OR price > upper BB (trailing)
            if close[i] < bb_middle[i] or close[i] > bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > middle BB OR price < lower BB (trailing)
            if close[i] > bb_middle[i] or close[i] < bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: BB breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > bb_upper[i] and close[i] > ema_50_aligned[i]:
                    # Break above upper BB with daily uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < bb_lower[i] and close[i] < ema_50_aligned[i]:
                    # Break below lower BB with daily downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals