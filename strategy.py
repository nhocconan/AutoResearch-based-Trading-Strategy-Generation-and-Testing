#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_HTF_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA200 as trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Supertrend on 6h data
    # ATR calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
            
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 10)  # warmup for weekly EMA200 and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(supertrend[i]) or 
            np.isnan(direction[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above Supertrend (uptrend) AND price above weekly EMA200 (bullish HTF)
            long_cond = (close[i] > supertrend[i]) and (close[i] > ema200_1w_aligned[i])
            
            # Short entry: price below Supertrend (downtrend) AND price below weekly EMA200 (bearish HTF)
            short_cond = (close[i] < supertrend[i]) and (close[i] < ema200_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below Supertrend (trend reversal)
            if close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above Supertrend (trend reversal)
            if close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Supertrend identifies trend direction and entry points on 6h timeframe, while weekly EMA200 acts as a higher-timeframe trend filter.
# Long when price is above Supertrend (6h uptrend) AND above weekly EMA200 (bullish long-term trend).
# Short when price is below Supertrend (6h downtrend) AND below weekly EMA200 (bearish long-term trend).
# This approach avoids counter-trend trading in strong markets and only takes trades aligned with the higher timeframe trend.
# Weekly EMA200 provides a robust, slow-moving filter that prevents whipsaws during sideways periods.
# Supertrend (with ATR=10, multiplier=3) adapts to volatility and provides clear trend signals.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.