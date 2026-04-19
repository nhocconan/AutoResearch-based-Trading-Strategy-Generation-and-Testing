#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12-hour Williams %R for momentum extremes, filtered by daily trend (EMA50) and volume confirmation.
# Williams %R identifies overbought/oversold conditions; we take counter-trend entries when price is also aligned with daily trend.
# In bull markets (price > daily EMA50), we look for oversold bounces (Williams %R < -80).
# In bear markets (price < daily EMA50), we look for overbought pullbacks (Williams %R > -20).
# Volume spike confirms institutional participation. Designed to work in both trending and ranging markets.
# Target: 20-30 trades/year per symbol with disciplined entries.
name = "6h_EMA50_1d_WilliamsR_12h_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12-hour Williams %R (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close']) / (highest_high - lowest_low)
    williams_r = williams_r.values  # convert to numpy array
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bull market bias: price above daily EMA50
            bull_market = close[i] > ema_50_1d_aligned[i]
            # Bear market bias: price below daily EMA50
            bear_market = close[i] < ema_50_1d_aligned[i]
            
            # Long: oversold in bull market OR oversold in bear market (counter-trend bounce)
            if ((bull_market and williams_r_aligned[i] < -80) or 
                (bear_market and williams_r_aligned[i] < -80)) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought in bear market OR overbought in bull market (counter-trend pullback)
            elif ((bear_market and williams_r_aligned[i] > -20) or 
                  (bull_market and williams_r_aligned[i] > -20)) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Williams %R becomes overbought or trend changes
            if williams_r_aligned[i] > -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Williams %R becomes oversold or trend changes
            if williams_r_aligned[i] < -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals