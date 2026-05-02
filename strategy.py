#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h EMA34 trend + volume confirmation
# Williams Alligator (jaw/teeth/lips) identifies trend absence when lines are intertwined.
# Entry on breakout when Alligator "awakens" (lines diverge) in direction of 12h EMA34 trend.
# Volume confirmation (1.8x 20-period average) reduces false signals.
# Works in both bull/bear markets by only taking trend-aligned breakouts when Alligator confirms trend strength.
# Discrete sizing 0.25 targets ~80-120 trades over 4 years (20-30/year).

name = "6h_WilliamsAlligator_12hEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Williams Alligator on 6h (primary timeframe)
    # Jaw: 13-period SMMA smoothed 8 periods ahead
    # Teeth: 8-period SMMA smoothed 5 periods ahead  
    # Lips: 5-period SMMA smoothed 3 periods ahead
    median = (high + low) / 2
    sma_13 = pd.Series(median).rolling(window=13, min_periods=13).mean()
    sma_8 = pd.Series(median).rolling(window=8, min_periods=8).mean()
    sma_5 = pd.Series(median).rolling(window=5, min_periods=5).mean()
    
    jaw = sma_13.rolling(window=8, min_periods=8).mean().shift(8).values
    teeth = sma_8.rolling(window=5, min_periods=5).mean().shift(5).values
    lips = sma_5.rolling(window=3, min_periods=3).mean().shift(3).values
    
    # Alligator lines divergence (trend strength): max - min of the three lines
    alligator_spread = np.maximum.reduce([jaw, teeth, lips]) - np.minimum.reduce([jaw, teeth, lips])
    alligator_spread_ma = pd.Series(alligator_spread).rolling(window=10, min_periods=10).mean().values
    
    # Alligator alignment: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
    alligator_long = (lips > teeth) & (teeth > jaw)
    alligator_short = (lips < teeth) & (teeth < jaw)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(alligator_spread_ma[i]) or 
            np.isnan(alligator_long[i]) or np.isnan(alligator_short[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator uptrend + price above Alligator + 12h uptrend + volume
            long_condition = (alligator_long[i] and 
                            close[i] > lips[i] and 
                            close[i] > ema_34_12h_aligned[i] and
                            volume_spike[i])
            
            # Short entry: Alligator downtrend + price below Alligator + 12h downtrend + volume
            short_condition = (alligator_short[i] and 
                             close[i] < lips[i] and 
                             close[i] < ema_34_12h_aligned[i] and
                             volume_spike[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses or price crosses below teeth
            if not alligator_long[i] or close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses or price crosses above teeth
            if not alligator_short[i] or close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals