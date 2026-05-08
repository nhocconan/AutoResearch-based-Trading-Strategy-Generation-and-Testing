#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WickReversal_Trend_Filter"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12-period ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    # Wick reversal detection: long when lower wick > 2x body and close > open
    # short when upper wick > 2x body and close < open
    body = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    open_ = prices['open'].values
    
    long_wick = (lower_wick > 2 * body) & (close > open_)
    short_wick = (upper_wick > 2 * body) & (close < open_)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(long_wick[i]) or np.isnan(short_wick[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish wick reversal + price above daily EMA34
            long_cond = long_wick[i] and (close[i] > ema_34_1d_aligned[i])
            
            # Short: bearish wick reversal + price below daily EMA34
            short_cond = short_wick[i] and (close[i] < ema_34_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA34 or ATR-based stop
            if (close[i] < ema_34_1d_aligned[i]) or (close[i] < ema_34_1d_aligned[i-1] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA34 or ATR-based stop
            if (close[i] > ema_34_1d_aligned[i]) or (close[i] > ema_34_1d_aligned[i-1] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals