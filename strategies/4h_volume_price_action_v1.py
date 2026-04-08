#!/usr/bin/env python3
"""
4h_volume_price_action_v1
Hypothesis: On 4h timeframe, use price action near swing highs/lows with volume confirmation and ATR filter. Enter long when price makes higher low with volume > 1.5x average; enter short when price makes lower high with volume > 1.5x average. Exit on opposite signal or ATR-based stop. Works in bull/bear via price action structure. Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_price_action_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Swing points: higher low and lower high
    # Higher low: current low > previous low AND previous low < low before that
    hl = np.zeros(n, dtype=bool)
    hh = np.zeros(n, dtype=bool)  # lower high: current high < previous high AND previous high > high before that
    
    for i in range(2, n):
        if low[i] > low[i-1] and low[i-1] < low[i-2]:
            hl[i] = True
        if high[i] < high[i-1] and high[i-1] > high[i-2]:
            hh[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or atr[i] <= 0 or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on short signal (lower high with volume)
            if hh[i] and vol_confirm:
                exit_long = True
            # Exit on ATR-based stop: price drops 2*ATR from entry
            # Track entry price implicitly through position holding
            elif i > 20 and close[i] < close[i-1] - 2.0 * atr[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on long signal (higher low with volume)
            if hl[i] and vol_confirm:
                exit_short = True
            # Exit on ATR-based stop: price rises 2*ATR from entry
            elif i > 20 and close[i] > close[i-1] + 2.0 * atr[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: higher low with volume confirmation
            long_entry = hl[i] and vol_confirm
            
            # Short entry: lower high with volume confirmation
            short_entry = hh[i] and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals