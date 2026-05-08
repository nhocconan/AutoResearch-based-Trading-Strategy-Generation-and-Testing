#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB (20,2) and close > 1d EMA(50), confirmed by volume spike
# Short when price breaks below lower BB (20,2) and close < 1d EMA(50), confirmed by volume spike
# Bollinger Bands adapt to volatility, working in both trending and ranging markets
# Target: 50-150 total trades over 4 years = 12-37/year

name = "4h_BollingerBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20,2) on 4h data
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma + 2 * std
    lower_bb = ma - 2 * std
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        upper_val = upper_bb[i]
        lower_val = lower_bb[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB + uptrend + volume spike
            if (close[i] > upper_val and 
                close[i] > ema50_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB + downtrend + volume spike
            elif (close[i] < lower_val and 
                  close[i] < ema50_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below middle band OR trend reverses
            if close[i] < ma[i] or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above middle band OR trend reverses
            if close[i] > ma[i] or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals