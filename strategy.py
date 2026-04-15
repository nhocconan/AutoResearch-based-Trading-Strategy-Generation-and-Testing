#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) and price > 1w EMA50 (bullish trend)
# Short when price breaks below lower BB(20,2) and price < 1w EMA50 (bearish trend)
# Uses volume > 1.5x 20-period median for confirmation
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns)
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20,2) on 1d
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate EMA50 on 1w
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            continue
        
        # Long entry: price breaks above upper BB + price > 1w EMA50 + volume confirmation
        if (close[i] > upper_bb_aligned[i] and
            close[i] > ema_50_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower BB + price < 1w EMA50 + volume confirmation
        elif (close[i] < lower_bb_aligned[i] and
              close[i] < ema_50_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse BB breakout or trend change
        elif position == 1 and (close[i] < lower_bb_aligned[i] or close[i] < ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper_bb_aligned[i] or close[i] > ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Breakout_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0