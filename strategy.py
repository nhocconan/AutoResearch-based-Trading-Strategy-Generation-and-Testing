#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_TailRisk_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Tail risk: price near extreme of range (top/bottom 10%)
    range_size = highest_high - lowest_low
    upper_zone = lowest_low + 0.9 * range_size  # top 10%
    lower_zone = lowest_low + 0.1 * range_size  # bottom 10%
    
    # Daily volatility filter: ATR(14) > 20-period SMA of ATR
    tr1 = high[1:] - low[:-1]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[:-1] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr14).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr14 > 1.5 * atr_ma20  # elevated volatility regime
    
    # Weekly trend filter: EMA50 on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr_ma20[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band + in upper zone + elevated vol + weekly uptrend
            if (price > highest_high[i] and  # Donchian breakout
                price > upper_zone[i] and    # in top 10% of range
                vol_filter[i] and            # volatility expansion
                price > ema50_1w_aligned[i]): # weekly uptrend
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below lower Donchian band + in lower zone + elevated vol + weekly downtrend
            elif (price < lowest_low[i] and   # Donchian breakdown
                  price < lower_zone[i] and   # in bottom 10% of range
                  vol_filter[i] and           # volatility expansion
                  price < ema50_1w_aligned[i]): # weekly downtrend
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price retreats to middle of range or weekly trend fails
            if (price < (highest_high[i] + lowest_low[i]) / 2 or  # retreat to mid-range
                price < ema50_1w_aligned[i]):                     # weekly trend fail
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises to middle of range or weekly trend fails
            if (price > (highest_high[i] + lowest_low[i]) / 2 or  # retreat to mid-range
                price > ema50_1w_aligned[i]):                     # weekly trend fail
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals