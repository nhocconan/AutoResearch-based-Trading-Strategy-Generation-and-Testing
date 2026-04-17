#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day SMA and standard deviation for Bollinger Bands
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + (2 * std20_1d)
    lower_bb_1d = sma20_1d - (2 * std20_1d)
    
    # Calculate 14-day RSI
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # Fill NaN with 50 (neutral)
    
    # Align indicators to 1d timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need RSI, Bollinger Bands, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: Price touches lower Bollinger Band AND RSI < 30 (oversold) with volume
            if (low[i] <= lower_bb_aligned[i] and rsi_aligned[i] < 30 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band AND RSI > 70 (overbought) with volume
            elif (high[i] >= upper_bb_aligned[i] and rsi_aligned[i] > 70 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price touches upper Bollinger Band OR RSI > 70
            if (high[i] >= upper_bb_aligned[i] or rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price touches lower Bollinger Band OR RSI < 30
            if (low[i] <= lower_bb_aligned[i] or rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BollingerRSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0