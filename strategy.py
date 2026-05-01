#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(21) pullback strategy with 4h trend filter and 1d volume confirmation.
# Long when: price pulls back to EMA(21) in 1h uptrend (4h EMA50 > EMA200) AND 1d volume > 1.5x 20-day average.
# Short when: price pulls back to EMA(21) in 1h downtrend (4h EMA50 < EMA200) AND 1d volume > 1.5x 20-day average.
# Uses discrete sizing 0.20 to manage drawdown. Target: 60-150 total trades over 4 years (15-37/year).
# 4h EMA crossover provides robust trend filter that works in both bull and bear markets.
# 1d volume confirmation ensures institutional participation, reducing false signals.
# 1h EMA(21) pullback provides precise entry timing with favorable risk-reward.

name = "1h_EMA21_Pullback_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 and EMA200 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = close_4h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe (wait for completed 4h bar)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume MA(20) for confirmation
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h EMA(21) for pullback entries
    close_s = pd.Series(close)
    ema21_1h = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08-20 UTC (already datetime64, use index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or \
           np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema21_1h[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema21 = ema21_1h[i]
        
        # Trend filter from 4h
        uptrend_4h = ema50_4h_aligned[i] > ema200_4h_aligned[i]
        downtrend_4h = ema50_4h_aligned[i] < ema200_4h_aligned[i]
        
        # Volume confirmation from 1d
        volume_confirm = vol_ma_1d_aligned[i] > 0 and volume[i] > (vol_ma_1d_aligned[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: pullback to EMA21 in 1h uptrend (4h) AND volume confirmation
            if (curr_low <= curr_ema21 <= curr_high and  # price touches EMA21
                uptrend_4h and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: pullback to EMA21 in 1h downtrend (4h) AND volume confirmation
            elif (curr_low <= curr_ema21 <= curr_high and  # price touches EMA21
                  downtrend_4h and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below EMA21 OR 4h trend turns down
            if (curr_close < curr_ema21 or 
                not uptrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above EMA21 OR 4h trend turns up
            if (curr_close > curr_ema21 or 
                not downtrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals