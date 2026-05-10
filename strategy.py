#!/usr/bin/env python3
# 1D_RSI50_Trend_With_W1_SupportResistance
# Hypothesis: Uses daily RSI(50) as trend filter with weekly support/resistance from prior weekly high/low.
# Enters long when RSI crosses above 50 and price breaks above prior weekly high with volume > 1.5x 20-day average.
# Enters short when RSI crosses below 50 and price breaks below prior weekly low with volume > 1.5x 20-day average.
# Exits when RSI crosses back below/above 50 or price returns to opposite weekly level.
# Uses RSI(50) as neutral trend filter to avoid whipsaws and works in both bull/bear markets.
# Targets 8-20 trades per year on daily timeframe with position size 0.25 to minimize fee drag.

name = "1D_RSI50_Trend_With_W1_SupportResistance"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for support/resistance levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly high and low (from completed weekly bar)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to daily timeframe (available after weekly bar closes)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate daily RSI(50) for trend filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/50, adjust=False, min_periods=50).mean()
    avg_loss = loss.ewm(alpha=1/50, adjust=False, min_periods=50).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for RSI
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # RSI trend filter
        rsi_above_50 = rsi_values[i] > 50
        rsi_below_50 = rsi_values[i] < 50
        rsi_crossed_up = rsi_values[i] > 50 and rsi_values[i-1] <= 50
        rsi_crossed_down = rsi_values[i] < 50 and rsi_values[i-1] >= 50
        
        if position == 0:
            # Long entry: RSI crosses above 50 and price breaks above weekly high with volume
            if (rsi_crossed_up and 
                close[i] > weekly_high_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI crosses below 50 and price breaks below weekly low with volume
            elif (rsi_crossed_down and 
                  close[i] < weekly_low_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses below 50 or price returns to weekly low
            if (rsi_crossed_down or 
                close[i] < weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses above 50 or price returns to weekly high
            if (rsi_crossed_up or 
                close[i] > weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals