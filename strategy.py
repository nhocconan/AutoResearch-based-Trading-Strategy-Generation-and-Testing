#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla levels
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    range_prev = prev_high - prev_low
    # Camarilla levels: R3, S3
    R3 = prev_close + 1.1 * range_prev / 2  # Resistance level 3
    S3 = prev_close - 1.1 * range_prev / 2  # Support level 3
    
    # Volume filter: today's volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    # Weekly trend filter: EMA34 on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(R3[i]) or np.isnan(S3[i]) or \
           np.isnan(vol_ma20[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume filter + weekly uptrend
            if (price > R3[i] and
                vol_filter[i] and
                price > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S3 + volume filter + weekly downtrend
            elif (price < S3[i] and
                  vol_filter[i] and
                  price < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price falls below S3 or weekly trend fails
            if (price < S3[i] or
                price < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R3 or weekly trend fails
            if (price > R3[i] or
                price > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals