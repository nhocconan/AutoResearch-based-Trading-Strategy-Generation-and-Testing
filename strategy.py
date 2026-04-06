#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Bollinger Band breakout with 1-day ATR filter and 1-week volume confirmation.
# In trending markets, price tends to stay near BB upper/lower bands during strong moves.
# We enter on BB breakouts only when 1-day ATR confirms momentum (ATR rising) and 
# weekly volume exceeds 20-day average (institutional participation).
# Works in bull/bear: BB breakouts capture momentum bursts in any direction.
# Target: 50-150 total trades over 4 years with strict entry filters.

name = "6h_bb_breakout_atr_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_length = 20
    bb_mult = 2.0
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(bb_length - 1, n):
        sma[i] = np.mean(close[i-bb_length+1:i+1])
        std[i] = np.std(close[i-bb_length+1:i+1])
        upper[i] = sma[i] + bb_mult * std[i]
        lower[i] = sma[i] - bb_mult * std[i]
    
    # 1-day ATR(14) for momentum confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.full(len(close_1d), np.nan)
    if len(close_1d) > 1:
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i],
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
    
    # ATR calculation
    atr_1d = np.full(len(close_1d), np.nan)
    atr_period = 14
    if len(close_1d) >= atr_period:
        atr_1d[atr_period-1] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    atr_ma_1d = np.full(len(close_1d), np.nan)
    ma_period = 10
    if len(close_1d) >= ma_period:
        for i in range(ma_period-1, len(close_1d)):
            atr_ma_1d[i] = np.mean(atr_1d[i-ma_period+1:i+1])
    
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    vol_lookback = 20
    if len(vol_1w) >= vol_lookback:
        for i in range(vol_lookback-1, len(vol_1w)):
            vol_ma_1w[i] = np.mean(vol_1w[i-vol_lookback+1:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(bb_length-1, atr_period+ma_period-2, vol_lookback-1)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(atr_ma_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses below middle band or stoploss
            if (close[i] < sma[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above middle band or stoploss
            if (close[i] > sma[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: BB breakout with volume and ATR confirmation
            bb_breakout_up = close[i] > upper[i] and close[i-1] <= upper[i-1]
            bb_breakout_down = close[i] < lower[i] and close[i-1] >= lower[i-1]
            
            # Volume filter: current volume > 1.5x weekly average
            volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
            
            # ATR filter: current ATR > 10-day average (momentum increasing)
            atr_filter = atr_ma_aligned[i] > 0 and not np.isnan(atr_ma_aligned[i])
            
            if volume_filter and atr_filter:
                if bb_breakout_up:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif bb_breakout_down:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals