#!/usr/bin/env python3
# 1d_MultiTimeframe_Structure_Trend
# Hypothesis: Combines 1-week structure (higher highs/lows) with 1-day trend and volume to capture multi-week moves.
# In bull markets: buy when price makes higher high (weekly structure) + price above daily EMA + volume spike.
# In bear markets: sell when price makes lower low (weekly structure) + price below daily EMA + volume spike.
# Uses structure to filter noise and trend/volume for confirmation. Targets 10-25 trades/year for low friction.

name = "1d_MultiTimeframe_Structure_Trend"
timeframe = "1d"
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
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly structure: higher highs and higher lows (uptrend), lower lows and lower highs (downtrend)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly swing points
    whh = np.maximum.accumulate(weekly_high)  # weekly higher high
    wll = np.minimum.accumulate(weekly_low)   # weekly lower low
    
    # Align weekly structure to daily
    whh_aligned = align_htf_to_ltf(prices, df_1w, whh)
    wll_aligned = align_htf_to_ltf(prices, df_1w, wll)
    
    # Get daily EMA for trend filter
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for daily EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(whh_aligned[i]) or np.isnan(wll_aligned[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d[i]
        downtrend = close[i] < ema_50_1d[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Weekly structure: price above weekly higher high = bullish structure
        bullish_structure = close[i] > whh_aligned[i]
        # Weekly structure: price below weekly lower low = bearish structure
        bearish_structure = close[i] < wll_aligned[i]
        
        if position == 0:
            # Long entry: bullish weekly structure + uptrend + volume spike
            if bullish_structure and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish weekly structure + downtrend + volume spike
            elif bearish_structure and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly lower low or trend reversal
            if close[i] < wll_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly higher high or trend reversal
            if close[i] > whh_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals