#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h momentum and 1d trend filter. Trade 4h EMA crossovers 
with 12h RSI momentum confirmation and 1d EMA200 trend filter. Use volume filter to 
avoid noise. Designed for fewer trades (target 20-40/year) to reduce fee drag and 
improve generalization. Works in bull via trend-following and in bear via momentum 
reversals at key levels.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA crossover signal
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(9) and EMA(21) for crossover
    ema9_4h = pd.Series(close_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get 12h data for RSI momentum
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI(14)
    delta = pd.Series(close_12h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = (100 - (100 / (1 + rs))).values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(200) for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all to 4h
    ema9_4h_aligned = align_htf_to_ltf(prices, df_4h, ema9_4h)
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema9_4h_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: EMA9 crosses above EMA21, RSI > 50 (bullish momentum), above 1d EMA200, with volume
            if (ema9_4h_aligned[i] > ema21_4h_aligned[i] and 
                ema9_4h_aligned[i-1] <= ema21_4h_aligned[i-1] and
                rsi_12h_aligned[i] > 50 and 
                close[i] > ema200_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: EMA9 crosses below EMA21, RSI < 50 (bearish momentum), below 1d EMA200, with volume
            elif (ema9_4h_aligned[i] < ema21_4h_aligned[i] and 
                  ema9_4h_aligned[i-1] >= ema21_4h_aligned[i-1] and
                  rsi_12h_aligned[i] < 50 and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: EMA9 crosses back below EMA21 or RSI drops below 40
            if (ema9_4h_aligned[i] < ema21_4h_aligned[i] and 
                ema9_4h_aligned[i-1] >= ema21_4h_aligned[i-1]) or \
               rsi_12h_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA9 crosses back above EMA21 or RSI rises above 60
            if (ema9_4h_aligned[i] > ema21_4h_aligned[i] and 
                ema9_4h_aligned[i-1] <= ema21_4h_aligned[i-1]) or \
               rsi_12h_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA9_21_RSI12h_EMA200d_Volume"
timeframe = "4h"
leverage = 1.0