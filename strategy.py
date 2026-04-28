#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (all lines intertwined) vs presence (diverged).
# Entry when Lips cross above/below Teeth with Jaw slope confirmation, aligned with 1d EMA50 trend.
# Volume confirmation ensures breakout validity. Discrete sizing (0.25) limits drawdown and fee churn.
# Designed to work in both bull (trend following) and bear (mean reversion during range) markets via regime filter.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h data (Smoothed Moving Average with specific periods)
    # Jaw: Blue line - 13-period SMMA, shifted 8 bars ahead
    # Teeth: Red line - 8-period SMMA, shifted 5 bars ahead  
    # Lips: Green line - 5-period SMMA, shifted 3 bars ahead
    # Using EMA as proxy for SMMA (similar smoothing effect)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Jaw slope: positive = bullish alignment, negative = bearish alignment
    jaw_slope = np.diff(jaw, prepend=jaw[0])
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 20)  # Max of Alligator periods and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        price = close[i]
        
        # Alligator signals: Lips cross above/below Teeth with Jaw slope confirmation
        lips_above_teeth = lips[i] > teeth[i]
        lips_below_teeth = lips[i] < teeth[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Lips cross above Teeth, bullish Jaw slope, above 1d EMA50, volume confirm
            if lips_above_teeth and jaw_slope[i] > 0 and price > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Lips cross below Teeth, bearish Jaw slope, below 1d EMA50, volume confirm
            elif lips_below_teeth and jaw_slope[i] < 0 and price < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on Lips cross below Teeth or below 1d EMA50
            if lips_below_teeth or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on Lips cross above Teeth or above 1d EMA50
            if lips_above_teeth or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals