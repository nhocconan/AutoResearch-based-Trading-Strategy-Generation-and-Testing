#!/usr/bin/env python3
name = "12H_WickReversal_1wTrend_Volume"
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
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    volume_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            volume_ma[i] = np.nan
        else:
            volume_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        uptrend = ema20_1w_aligned[i] > ema20_1w_aligned[i-1]
        downtrend = ema20_1w_aligned[i] < ema20_1w_aligned[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Wick rejection signals
        body_size = abs(close[i] - open_price[i])
        total_range = high[i] - low[i]
        upper_wick = high[i] - max(close[i], open_price[i])
        lower_wick = min(close[i], open_price[i]) - low[i]
        
        # Bullish reversal: long lower wick, small body, in uptrend
        bullish_rejection = (lower_wick > body_size * 2) and (body_size < total_range * 0.3)
        # Bearish rejection: long upper wick, small body, in downtrend
        bearish_rejection = (upper_wick > body_size * 2) and (body_size < total_range * 0.3)
        
        if position == 0:
            # Enter long: bullish rejection + uptrend + volume
            if bullish_rejection and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish rejection + downtrend + volume
            elif bearish_rejection and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish rejection or trend change
            if bearish_rejection or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish rejection or trend change
            if bullish_rejection or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals