#!/usr/bin/env python3
name = "4h_KAMA_Trend_With_Volume_And_Chop"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR for chop filter
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(abs(high_1d - np.roll(close_1d, 1)), abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Bollinger Bands for chop filter
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # 4h KAMA
    change = np.abs(np.subtract(close[4:], close[:-4]))
    volatility = np.sum(np.abs(np.diff(close.reshape(-1, 1), axis=0).reshape(-1)[4:].reshape(-1, 4)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = np.power(er * (2/(2+1) - 2/(30+1)) + 2/(30+1), 2)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(bb_width_1d_aligned[i]) or np.isnan(kama[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Chop filter: choppy market when BB width is high
        is_choppy = bb_width_1d_aligned[i] > 0.05
        
        if position == 0:
            # Long: price above KAMA, uptrend (price > EMA34), not choppy
            if (close[i] > kama[i] and 
                close[i] > ema34_1d_aligned[i] and
                not is_choppy):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, downtrend (price < EMA34), not choppy
            elif (close[i] < kama[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  not is_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or choppy market
            if (close[i] < kama[i] or is_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA or choppy market
            if (close[i] > kama[i] or is_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals