#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 12h EMA Trend Filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA)
# - Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) and 12h EMA34 > 12h EMA89 (uptrend)
# - Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) and 12h EMA34 < 12h EMA89 (downtrend)
# - Combines momentum strength with trend filter to avoid whipsaws in ranging markets
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 and EMA89 on 12h timeframe
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    # Calculate EMA13 for Elder Ray on 6h timeframe
    close_6h = prices['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema13_6h  # High - EMA13
    bear_power = ema13_6h - low_6h   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if NaN in indicators
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(ema89_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull = bull_power[i]
        bear = bear_power[i]
        ema34 = ema34_12h_aligned[i]
        ema89 = ema89_12h_aligned[i]
        
        if position == 0:
            # Long entry: Bull Power > 0 AND Bear Power < 0 (bullish momentum) 
            # AND 12h EMA34 > EMA89 (uptrend)
            if bull > 0 and bear < 0 and ema34 > ema89:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 AND Bull Power < 0 (bearish momentum)
            # AND 12h EMA34 < EMA89 (downtrend)
            elif bear > 0 and bull < 0 and ema34 < ema89:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Momentum deteriorates or trend turns bearish
            if bull <= 0 or bear >= 0 or ema34 <= ema89:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Momentum deteriorates or trend turns bullish
            if bear <= 0 or bull >= 0 or ema34 >= ema89:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA_TrendFilter"
timeframe = "6h"
leverage = 1.0