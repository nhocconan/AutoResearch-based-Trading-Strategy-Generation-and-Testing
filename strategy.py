# 1d_1w_FVG_Trend_V1 - Fair Value Gap strategy with weekly trend filter
# Hypothesis: 1d timeframe using Fair Value Gaps (FVG) from price inefficiencies
# combined with 1-week EMA trend filter for direction. Enters when price
# retraces to fill a FVG in the direction of the weekly trend.
# Targets 10-25 trades/year (40-100 total over 4 years) with selective entries.
# Works in bull/bear by following higher timeframe trend direction.
# Uses volume confirmation to avoid low-quality signals.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_FVG_Trend_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for FVG detection (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    # Detect FVG (Fair Value Gap) - 3-bar pattern with gap
    # Bullish FVG: gap between low[i-2] and high[i] where low[i] > high[i-2]
    # Bearish FVG: gap between high[i-2] and low[i] where high[i] < low[i-2]
    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Bullish FVG: gap up - inefficient move up
        if low[i] > high[i-2]:
            bullish_fvg[i] = True
        # Bearish FVG: gap down - inefficient move down  
        if high[i] < low[i-2]:
            bearish_fvg[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price retraces to fill bullish FVG AND above weekly EMA50 with volume
            if (bullish_fvg[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price retraces to fill bearish FVG AND below weekly EMA50 with volume
            elif (bearish_fvg[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly EMA50 or fills the gap completely
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly EMA50 or fills the gap completely
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals