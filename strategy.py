#!/usr/bin/env python3
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
    
    # Get daily data for daily range and ATR
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily range (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    daily_range = high_1d - low_1d
    
    # Calculate 6h ATR(14) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 6-period 6h ATR average for volatility filter
    atr_ma_6 = np.full(n, np.nan)
    for i in range(6, n):
        atr_ma_6[i] = np.mean(atr[i-6:i])
    
    # Align daily range to 6h timeframe
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need ATR MA
    start_idx = 6
    
    for i in range(start_idx, n):
        if np.isnan(daily_range_aligned[i]) or np.isnan(atr_ma_6[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        daily_range_val = daily_range_aligned[i]
        
        # Volatility filter: ATR > 30% of daily range
        vol_filter = atr[i] > daily_range_val * 0.3
        
        # Dynamic breakout levels: 20% of daily range from previous close
        if i > 0:
            prev_close = close[i-1]
            upper_break = prev_close + daily_range_val * 0.2
            lower_break = prev_close - daily_range_val * 0.2
        else:
            upper_break = lower_break = price
        
        if position == 0:
            # Long: break above upper level with volatility
            if vol_filter and price > upper_break:
                signals[i] = 0.25
                position = 1
            # Short: break below lower level with volatility
            elif vol_filter and price < lower_break:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to previous close or volatility drops
            if price < close[i-1] or atr[i] < daily_range_val * 0.15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to previous close or volatility drops
            if price > close[i-1] or atr[i] < daily_range_val * 0.15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyRangeBreakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0