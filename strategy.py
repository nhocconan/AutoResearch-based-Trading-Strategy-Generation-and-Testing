#!/usr/bin/env python3
"""
6h_cci_breakout_1d_trend_volume_v1
Hypothesis: On 6h timeframe, use CCI(20) breakout with daily trend filter and volume confirmation. 
Enter long when CCI crosses above +100 with daily EMA50 > EMA200 and volume > 1.5x average; 
enter short when CCI crosses below -100 with daily EMA50 < EMA200 and volume > 1.5x average. 
Exit when CCI crosses back through zero or trend reverses. 
CCI captures momentum extremes, daily EMA filter ensures trend alignment, volume confirms institutional participation.
Works in bull/bear via daily trend filter. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA50 and EMA200 for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate CCI(20) on 6h data
    typical_price = (high + low + close) / 3
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - tp_mean) / (0.015 * tp_mad)
    # Handle division by zero
    cci = np.where(tp_mad == 0, 0, cci)
    
    # Volume confirmation (24-period average on 6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(cci[i]) or np.isnan(ema50[i]) or np.isnan(ema200[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if CCI crosses below zero (momentum fade)
            if cci[i] < 0 and cci[i-1] >= 0:
                exit_long = True
            # Exit if daily EMA50 crosses below EMA200 (trend reversal)
            elif ema50[i] < ema200[i] and ema50[i-1] >= ema200[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if CCI crosses above zero (momentum fade)
            if cci[i] > 0 and cci[i-1] <= 0:
                exit_short = True
            # Exit if daily EMA50 crosses above EMA200 (trend reversal)
            elif ema50[i] > ema200[i] and ema50[i-1] <= ema200[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses above +100 with daily EMA50 > EMA200 and volume confirmation
            long_entry = False
            if (cci[i] > 100 and cci[i-1] <= 100 and
                ema50[i] > ema200[i] and vol_confirm):
                long_entry = True
            
            # Short entry: CCI crosses below -100 with daily EMA50 < EMA200 and volume confirmation
            short_entry = False
            if (cci[i] < -100 and cci[i-1] >= -100 and
                ema50[i] < ema200[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals