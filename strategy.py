#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Index_Mean_Reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index on daily data
    atr_period = 14
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum()
    highest_high = pd.Series(df_1d['high']).rolling(window=atr_period, min_periods=atr_period).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=atr_period, min_periods=atr_period).min()
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(atr_period)
    chop_values = chop.values
    
    # Get 1d SMA for trend context
    sma_50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    
    # Get 4h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all to 4h
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_values)
    sma_50_1d_4h = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(60, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(chop_4h[i]) or np.isnan(sma_50_1d_4h[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_4h[i]
        sma_50 = sma_50_1d_4h[i]
        rsi_val = rsi_values[i]
        
        if position == 0:
            # Enter long: choppy market (range) + RSI oversold + price below SMA
            if chop_val > 61.8 and rsi_val < 30 and close[i] < sma_50:
                signals[i] = 0.25
                position = 1
            # Enter short: choppy market (range) + RSI overbought + price above SMA
            elif chop_val > 61.8 and rsi_val > 70 and close[i] > sma_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or chop drops (trending)
            if rsi_val > 70 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or chop drops (trending)
            if rsi_val < 30 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals