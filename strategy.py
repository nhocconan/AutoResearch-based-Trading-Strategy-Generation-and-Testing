#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Long when Williams %R crosses above -80 from below (oversold bounce) with 1d EMA50 uptrend and volume > 1.5x average
# Short when Williams %R crosses below -20 from above (overbought rejection) with 1d EMA50 downtrend and volume > 1.5x average
# Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short)
# Williams %R identifies momentum extremes; 1d EMA50 filters trend direction; volume confirms conviction
# Designed to work in both bull (buy dips) and bear (sell rallies) markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_WilliamsR_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 14-period Williams %R
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = willr.replace([np.inf, -np.inf], np.nan).fillna(-50).values  # Handle division by zero
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Williams %R and EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below, EMA50 uptrend, volume spike
            if (willr[i] > -80 and willr[i-1] <= -80 and  # Cross above -80
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from above, EMA50 downtrend, volume spike
            elif (willr[i] < -20 and willr[i-1] >= -20 and  # Cross below -20
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -20 (overbought)
            if willr[i] < -20 and willr[i-1] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -80 (oversold)
            if willr[i] > -80 and willr[i-1] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals