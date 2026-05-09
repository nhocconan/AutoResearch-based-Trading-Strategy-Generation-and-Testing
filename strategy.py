#!/usr/bin/env python3
# Hypothesis: 6h price action with 1d Bollinger Bands width regime filter and 12h momentum confirmation
# Long when price > 12h EMA(20), BB width > 0.05 (volatile regime), and close > open (bullish candle)
# Short when price < 12h EMA(20), BB width > 0.05, and close < open (bearish candle)
# Exit when BB width < 0.03 (low volatility) or price crosses 12h EMA(20)
# Uses BB width to identify volatile regimes where breakouts are more likely to follow through
# EMA(20) provides dynamic support/resistance in trending markets
# Designed to capture momentum moves in volatile conditions while avoiding low-volume consolidations
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_BBWidth_Momentum_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Bollinger Bands for 1d timeframe (20-period, 2 std dev)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for BB calculation
        return np.zeros(n)
    
    # Typical price for BB calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Calculate SMA and standard deviation
    sma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    std_dev = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    
    # Upper and lower bands
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    
    # Bollinger Band Width: (Upper - Lower) / Middle
    bb_width = (upper_band - lower_band) / sma
    bb_width = np.where(sma != 0, bb_width, 0)  # Avoid division by zero
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate EMA(20) for 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_width_aligned[i]) or np.isnan(ema_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above EMA, volatile regime (BB width > 0.05), bullish candle
            if (close[i] > ema_20_12h_aligned[i] and 
                bb_width_aligned[i] > 0.05 and 
                close[i] > open_price[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA, volatile regime (BB width > 0.05), bearish candle
            elif (close[i] < ema_20_12h_aligned[i] and 
                  bb_width_aligned[i] > 0.05 and 
                  close[i] < open_price[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: low volatility (BB width < 0.03) or price crosses below EMA
            if (bb_width_aligned[i] < 0.03) or (close[i] < ema_20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: low volatility (BB width < 0.03) or price crosses above EMA
            if (bb_width_aligned[i] < 0.03) or (close[i] > ema_20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals