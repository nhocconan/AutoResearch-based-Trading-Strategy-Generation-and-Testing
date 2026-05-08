#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WickReversal_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-day average (avoid low-volume noise)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Wick Reversal signal:
    # Bullish: long lower wick > 2x body and close > open
    # Bearish: long upper wick > 2x body and close < open
    body = np.abs(close - open_)
    lower_wick = np.where(close >= open_, open_ - low, close - low)
    upper_wick = np.where(close >= open_, high - close, high - open_)
    
    bullish_wick = (lower_wick > 2 * body) & (close > open_)
    bearish_wick = (upper_wick > 2 * body) & (close < open_)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for weekly EMA20
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish wick + above weekly EMA + volume filter
            long_cond = bullish_wick[i] and (close[i] > ema_20_1w_aligned[i]) and volume_filter[i]
            # Short: bearish wick + below weekly EMA + volume filter
            short_cond = bearish_wick[i] and (close[i] < ema_20_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish wick appears or price crosses below weekly EMA
            if bearish_wick[i] or (close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish wick appears or price crosses above weekly EMA
            if bullish_wick[i] or (close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals