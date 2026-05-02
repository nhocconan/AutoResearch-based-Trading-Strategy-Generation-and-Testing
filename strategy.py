#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for HTF trend alignment to reduce whipsaw vs shorter trends
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 12h provides trend direction and entry signals
# Breakout alignment with 1d EMA50 ensures medium-term trend coherence
# Volume confirmation filters low-quality breakouts
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Discrete position sizing: 0.25 (25% of capital) to minimize fee churn
# Works in both bull and bear markets by following 1d trend with Alligator signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Alligator components: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price_12h = (high_12h + low_12h) / 2.0
    jaw_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # Jaw
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values   # Teeth
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values    # Lips
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 2.0x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(13, 20, 50)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator sleeping (jaws, teeth, lips intertwined) -> wait for awakening
            # Alligator awakening: lips > teeth > jaw (bullish) OR lips < teeth < jaw (bearish)
            # Long entry: bullish alignment AND price > jaw with volume spike AND price > 1d EMA50 (bullish trend)
            if (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and
                close[i] > jaw_12h_aligned[i] and
                volume_spike[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment AND price < jaw with volume spike AND price < 1d EMA50 (bearish trend)
            elif (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i] and
                  close[i] < jaw_12h_aligned[i] and
                  volume_spike[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator starts sleeping again (lips cross below teeth) OR price < jaw OR price < 1d EMA50
            if (lips_12h_aligned[i] < teeth_12h_aligned[i] or
                close[i] < jaw_12h_aligned[i] or
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator starts sleeping again (lips cross above teeth) OR price > jaw OR price > 1d EMA50
            if (lips_12h_aligned[i] > teeth_12h_aligned[i] or
                close[i] > jaw_12h_aligned[i] or
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals