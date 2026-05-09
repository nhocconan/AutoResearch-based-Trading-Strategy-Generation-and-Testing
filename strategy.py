#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Long when Williams %R crosses above -20 (exit oversold) + price above 1d EMA50 + volume spike
# Short when Williams %R crosses below -80 (enter overbought) + price below 1d EMA50 + volume spike
# Williams %R identifies mean reversion entries in ranging markets, while 1d EMA50 filters for trend direction
# Volume spike confirms institutional participation. Designed for 60-120 trades/year on BTC/ETH.

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
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
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = willr.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when no range
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -20 (exiting oversold) + uptrend + volume
            if (willr[i] > -20 and willr[i-1] <= -20 and  # crossover up
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -80 (entering overbought) + downtrend + volume
            elif (willr[i] < -80 and willr[i-1] >= -80 and  # crossover down
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -80 (overbought) OR trend fails
            if (willr[i] < -80 and willr[i-1] >= -80) or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -20 (oversold) OR trend fails
            if (willr[i] > -20 and willr[i-1] <= -20) or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals