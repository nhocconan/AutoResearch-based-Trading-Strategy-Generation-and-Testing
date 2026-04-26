#!/usr/bin/env python3
"""
6h_WilliamsAlligator_TrendWithWeeklyFilter
Hypothesis: Williams Alligator (SMAs with offsets) identifies trending regimes on 6h, while 1-week trend filter ensures alignment with higher-timeframe direction. 
In bull markets: price above Alligator lips/teeth/jaw with 1w uptrend → long. 
In bear markets: price below Alligator components with 1w downtrend → short. 
Uses volume confirmation to avoid choppy breakouts. Target: 50-150 trades over 4 years. 
Alligator's smoothed SMAs reduce whipsaw vs plain EMA crossovers.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need warmup for SMAs
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 1.5)
    
    # Williams Alligator on 6h: SMAs with offsets
    # Jaw: 13-period SMA, offset 8 bars
    # Teeth: 8-period SMA, offset 5 bars  
    # Lips: 5-period SMA, offset 3 bars
    close_series = pd.Series(close)
    jaw_raw = close_series.rolling(window=13, min_periods=13).mean().shift(8)
    teeth_raw = close_series.rolling(window=8, min_periods=8).mean().shift(5)
    lips_raw = close_series.rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw = jaw_raw.values
    teeth = teeth_raw.values
    lips = lips_raw.values
    
    # 1-week trend filter: EMA50 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 13+8=21 for jaw, plus 1w EMA)
    start_idx = 30
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_1w_val = ema_50_1w_aligned[i]
        
        # Alligator alignment: check if all three lines are ordered
        # Bullish alignment: lips > teeth > jaw (price above all)
        # Bearish alignment: lips < teeth < jaw (price below all)
        bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Long logic: bullish alignment + price above lips + 1w uptrend + volume spike
        long_condition = bullish_alignment and (close_val > lips_val) and (close_val > ema_1w_val) and volume_spike[i]
        # Short logic: bearish alignment + price below lips + 1w downtrend + volume spike
        short_condition = bearish_alignment and (close_val < lips_val) and (close_val < ema_1w_val) and volume_spike[i]
        
        # Exit logic: Alligator alignment breaks or 1w trend reversal
        exit_long = not bullish_alignment or (close_val < ema_1w_val)
        exit_short = not bearish_alignment or (close_val > ema_1w_val)
        
        # Minimum holding period: 2 bars
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WilliamsAlligator_TrendWithWeeklyFilter"
timeframe = "6h"
leverage = 1.0