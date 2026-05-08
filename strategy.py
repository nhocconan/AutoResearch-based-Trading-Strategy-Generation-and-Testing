#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h/1d trend filter and volume confirmation
# Uses 4h EMA50 for trend direction and 1d Bollinger Bands for mean reversion zones
# Entry: price touches 1d Bollinger lower band (for long) or upper band (for short)
# Only trade in direction of 4h EMA50 trend with volume confirmation
# Designed for low-frequency, high-conviction trades to minimize fee drag
# Works in both bull and bear markets via trend alignment and mean reversion

name = "1h_BollingerMeanRev_4hEMA50_1dVol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 20-period SMA and standard deviation for Bollinger Bands
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + (2 * std20_1d)
    lower_bb_1d = sma20_1d - (2 * std20_1d)
    
    # Align Bollinger Bands to 1h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Volume spike (1.5x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches lower Bollinger Band with 4h uptrend and volume spike
            if (close[i] <= lower_bb_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price touches upper Bollinger Band with 4h downtrend and volume spike
            elif (close[i] >= upper_bb_aligned[i] and 
                  close[i] < ema50_4h_aligned[i] and vol_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses above 4h EMA50 or touches upper Bollinger Band
            if (close[i] >= ema50_4h_aligned[i] or 
                close[i] >= upper_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses below 4h EMA50 or touches lower Bollinger Band
            if (close[i] <= ema50_4h_aligned[i] or 
                close[i] <= lower_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals