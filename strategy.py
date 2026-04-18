#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction and 1d ADX for trend strength filter.
# Enters on 1h pullbacks to EMA21 in the direction of 4h Supertrend when 1d ADX > 25.
# Uses tight stops and targets to limit trades (target 15-30/year) and avoid fee drag.
# Designed to work in both bull (trend following) and bear (counter-trend at extremes) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = high_4h[0] - low_4h[0]
    tr2[0] = np.abs(high_4h[0] - close_4h[0])
    tr3[0] = np.abs(low_4h[0] - close_4h[0])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    basic_ub = (high_4h + low_4h) / 2 + 3.0 * atr_4h
    basic_lb = (high_4h + low_4h) / 2 - 3.0 * atr_4h
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if basic_ub[i] < final_ub[i-1] or close_4h[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_4h[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    for i in range(len(close_4h)):
        if i == 0:
            supertrend[i] = final_lb[i]
            direction[i] = 1
        elif supertrend[i-1] == final_ub[i-1]:
            if close_4h[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
                direction[i] = -1
        else:
            if close_4h[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
                direction[i] = 1
    
    # Align 4h Supertrend direction to 1h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1h EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 30)  # need EMA21 and enough ADX data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(open_price[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long entry: pullback to EMA21 in uptrend
            if (supertrend_dir_aligned[i] == 1 and 
                strong_trend and
                low[i] <= ema_21[i] <= high[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: pullback to EMA21 in downtrend
            elif (supertrend_dir_aligned[i] == -1 and 
                  strong_trend and
                  low[i] <= ema_21[i] <= high[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: trend change or ADX weakens
            if (supertrend_dir_aligned[i] == -1 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend change or ADX weakens
            if (supertrend_dir_aligned[i] == 1 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Supertrend4h_ADX1d_EMA21_Pullback"
timeframe = "1h"
leverage = 1.0