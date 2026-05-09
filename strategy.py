#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Weekly OHLC from previous week (for Camarilla)
    prev_high_w = np.roll(high, 1)
    prev_low_w = np.roll(low, 1)
    prev_close_w = np.roll(close, 1)
    prev_open_w = np.roll(prices['open'].values, 1)
    prev_high_w[0] = high[0]
    prev_low_w[0] = low[0]
    prev_close_w[0] = close[0]
    prev_open_w[0] = prices['open'].values[0]
    
    # Calculate Weekly Camarilla R3/S3 levels
    range_w = prev_high_w - prev_low_w
    close_prev_w = prev_close_w
    r3 = close_prev_w + range_w * 1.1 / 4
    s3 = close_prev_w - range_w * 1.1 / 4
    
    # Weekly trend: EMA50 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.3x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r3[i]) or np.isnan(s3[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above R3 with weekly uptrend and volume
            if (price > r3[i] and 
                price > ema50_1w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: breakdown below S3 with weekly downtrend and volume
            elif (price < s3[i] and 
                  price < ema50_1w_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to weekly EMA or loses volume
            if (price < ema50_1w_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly EMA or loses volume
            if (price > ema50_1w_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals