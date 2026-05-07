#!/usr/bin/env python3
name = "6h_ElderRay_Ray_Crossover_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Elder Ray components from daily data
    # Bull Power = daily high - EMA(13) of daily close
    # Bear Power = daily low - EMA(13) of daily close
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34)  # Wait for EMA(13) and EMA(34)
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero with 12h uptrend
            bull_cross_up = bull_power_aligned[i] > 0 and bull_power_aligned[i-1] <= 0
            uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]
            
            if bull_cross_up and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses below zero with 12h downtrend
            elif bear_power_aligned[i] < 0 and bear_power_aligned[i-1] >= 0:
                downtrend = ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]
                if downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: Bull Power crosses below zero or trend turns down
            if bull_power_aligned[i] < 0 or ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power crosses above zero or trend turns up
            if bear_power_aligned[i] > 0 or ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray crossover with 12h trend filter
# - Elder Ray (Bull Power/Bear Power) measures daily buying/selling pressure relative to EMA(13)
# - Long when Bull Power crosses above zero in 12h uptrend (bullish momentum with trend)
# - Short when Bear Power crosses below zero in 12h downtrend (bearish momentum with trend)
# - Works in both bull (buy signals in uptrend) and bear (sell signals in downtrend) markets
# - Exit when power crosses zero or trend reverses
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Uses daily Elder Ray for institutional pressure, 12h EMA for trend filter
# - Novel combination: Elder Ray (1d) + trend (12h) not recently tried on 6h
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits