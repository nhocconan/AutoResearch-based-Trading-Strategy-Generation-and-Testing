#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_4hrCloseAbovePrevClose_SimpleTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 3:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily trend filter: close > previous day close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_prev_close = np.roll(daily_close, 1)
    daily_prev_close[0] = np.nan
    daily_uptrend = daily_close > daily_prev_close
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    
    # 4h volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(daily_uptrend_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        is_uptrend = daily_uptrend_aligned[i]
        
        if position == 0:
            # Enter long: 4h close > previous 4h close, volume confirmation, and daily uptrend
            if close[i] > close[i-1] and vol > 1.5 * vol_ma_val and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: 4h close < previous 4h close, volume confirmation, and daily downtrend
            elif close[i] < close[i-1] and vol > 1.5 * vol_ma_val and not is_uptrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 4h close < previous 4h close or daily trend turns down
            if close[i] < close[i-1] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 4h close > previous 4h close or daily trend turns up
            if close[i] > close[i-1] or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals