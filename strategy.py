# 6h_SuperTrend_TripleFilter
# Hypothesis: SuperTrend on 6h with EMA200 trend filter and volume confirmation captures strong trends while avoiding whipsaws.
# Works in bull: rides uptrends with SuperTrend long signals.
# Works in bear: avoids false longs in downtrends via EMA200 filter and takes shorts when SuperTrend flips.
# Volume filter ensures breakouts have conviction. Target: 20-40 trades/year on 6h timeframe.
# Uses 1d EMA200 and volume MA20, aligned to 6h chart. No look-ahead bias.

#!/usr/bin/env python3
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
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(200) for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate SuperTrend on 6h data
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR
    atr = np.zeros_like(close)
    atr[:atr_period] = np.nan
    atr[atr_period] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period+1, len(close)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close)):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # SuperTrend
    supertrend = np.zeros_like(close)
    supertrend[0] = final_ub[0]
    direction = np.ones_like(close, dtype=int)  # 1 for uptrend, -1 for downtrend
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > final_ub[i-1]:
            direction[i] = 1
        elif close[i] < final_lb[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lb[i]
        else:
            supertrend[i] = final_ub[i]
    
    # Align daily indicators to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_aligned[i]) or 
            np.isnan(supertrend[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # SuperTrend signals
        supertrend_long = direction[i] == 1
        supertrend_short = direction[i] == -1
        
        # Entry conditions: SuperTrend signal with trend and volume filter
        long_entry = supertrend_long and uptrend and vol_filter
        short_entry = supertrend_short and downtrend and vol_filter
        
        # Exit conditions: SuperTrend reversal or trend change
        long_exit = not supertrend_long or not uptrend
        short_exit = not supertrend_short or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_SuperTrend_TripleFilter"
timeframe = "6h"
leverage = 1.0