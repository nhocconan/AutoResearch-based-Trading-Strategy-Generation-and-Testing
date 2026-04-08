#!/usr/bin/env python3
# 1d_moving_average_crossover_1w_trend_v1
# Hypothesis: On 1d timeframe, use EMA crossover (21/50) with 1w trend filter and volume confirmation.
# Long when EMA21 crosses above EMA50 with volume > 1.5x average and 1w uptrend.
# Short when EMA21 crosses below EMA50 with volume > 1.5x average and 1w downtrend.
# Exit when opposite crossover occurs.
# Uses daily EMA crossover for trend changes, weekly trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year to minimize fee drag while capturing major trend shifts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_moving_average_crossover_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA21 and EMA50
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1w trend filter: EMA21 on weekly data
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema21)
    
    # Volume confirmation: 20-period average on 1d
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema21[i]) or np.isnan(ema50[i]) or np.isnan(weekly_ema21_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA21 crosses below EMA50
            if ema21[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA21 crosses above EMA50
            if ema21[i] > ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Weekly trend filter
            weekly_uptrend = close[i] > weekly_ema21_aligned[i]
            weekly_downtrend = close[i] < weekly_ema21_aligned[i]
            
            # Bullish crossover: EMA21 crosses above EMA50
            bullish_crossover = ema21[i] > ema50[i] and ema21[i-1] <= ema50[i-1]
            # Bearish crossover: EMA21 crosses below EMA50
            bearish_crossover = ema21[i] < ema50[i] and ema21[i-1] >= ema50[i-1]
            
            # Long entry: bullish crossover with volume and weekly uptrend
            if bullish_crossover and volume_ok and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish crossover with volume and weekly downtrend
            elif bearish_crossover and volume_ok and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals