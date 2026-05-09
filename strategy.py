#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_AggressiveBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for breakout levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Aggressive breakout: 1.5x daily range
    range_ = prev_high - prev_low
    upper_break = prev_close + range_ * 1.5
    lower_break = prev_close - range_ * 1.5
    
    # Daily trend: EMA34 on 1d (one-bar delay via align)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 30-period SMA
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.5 * vol_ma30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(upper_break[i]) or np.isnan(lower_break[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: aggressive breakout above upper band with daily uptrend and volume
            if (price > upper_break[i] and 
                price > ema34_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: aggressive breakdown below lower band with daily downtrend and volume
            elif (price < lower_break[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to daily EMA or loses volume
            if (price < ema34_1d_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to daily EMA or loses volume
            if (price > ema34_1d_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals