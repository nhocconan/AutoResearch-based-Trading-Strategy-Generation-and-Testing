#!/usr/bin/env python3
"""
4H_PriceAction_12hTrend_Volume_Signal
Hypothesis: Uses price action (Higher Highs/Higher Lows) for trend identification combined with 
12h EMA trend filter and volume confirmation. Designed for 4h timeframe to capture 
trend continuation moves with low trade frequency (target: 20-40 trades/year). 
Works in both bull and bear markets by following 12h trend direction, avoiding 
counter-trend trades. Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "4H_PriceAction_12hTrend_Volume_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Price action: Higher Highs and Higher Lows for uptrend, Lower Highs and Lower Lows for downtrend
    # Using 3-period lookback for swing points
    hh = np.zeros(n, dtype=bool)  # Higher High
    ll = np.zeros(n, dtype=bool)  # Higher Low
    lh = np.zeros(n, dtype=bool)  # Lower High
    ll_pattern = np.zeros(n, dtype=bool)  # Lower Low
    
    for i in range(2, n):
        # Higher High: current high > previous high and previous high > high before that
        hh[i] = (high[i] > high[i-1]) and (high[i-1] > high[i-2])
        # Higher Low: current low > previous low and previous low > low before that
        ll[i] = (low[i] > low[i-1]) and (low[i-1] > low[i-2])
        # Lower High: current high < previous high and previous high < high before that
        lh[i] = (high[i] < high[i-1]) and (high[i-1] < high[i-2])
        # Lower Low: current low < previous low and previous low < low before that
        ll_pattern[i] = (low[i] < low[i-1]) and (low[i-1] < low[i-2])
    
    # Uptrend: HH and HL, Downtrend: LH and LL
    uptrend = hh & ll
    downtrend = lh & ll_pattern
    
    # Volume filter: volume > 1.5x 20-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Long entry: uptrend + price above 12h EMA + volume spike
            if (uptrend[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price below 12h EMA + volume spike
            elif (downtrend[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: downtrend begins or volume drops
            if (downtrend[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: uptrend begins or volume drops
            if (uptrend[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals