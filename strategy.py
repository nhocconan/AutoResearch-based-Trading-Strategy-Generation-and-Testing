#!/usr/bin/env python3
"""
12h_roc_volume_breakout_1w_trend_v1
Hypothesis: On 12h timeframe, use Rate of Change (ROC) for momentum strength with 1-week EMA for trend filter and volume confirmation. Enter long when ROC crosses above zero with price above EMA and volume confirmation; enter short when ROC crosses below zero with price below EMA and volume confirmation. Exit when ROC crosses back to zero or opposite extreme. This strategy targets momentum shifts with volume confirmation, reducing false signals and trade frequency. Works in bull/bear via trend filter and breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_roc_volume_breakout_1w_trend_v1"
timeframe = "12h"
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
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA on 1w data
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate ROC on 12h (10-period)
    roc = np.zeros(n)
    for i in range(10, n):
        if close[i-10] != 0:
            roc[i] = (close[i] - close[i-10]) / close[i-10] * 100
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_12h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(roc[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_1w_12h[i]
        downtrend = close[i] < ema_1w_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if ROC crosses back below zero (momentum fading)
            if roc[i] < 0 and roc[i-1] >= 0:
                exit_long = True
            # Exit if trend turns down
            elif downtrend and roc[i] < 0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if ROC crosses back above zero (momentum fading)
            if roc[i] > 0 and roc[i-1] <= 0:
                exit_short = True
            # Exit if trend turns up
            elif uptrend and roc[i] > 0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # ROC crosses above zero with uptrend and volume confirmation
            if roc[i] > 0 and roc[i-1] <= 0:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # ROC crosses below zero with downtrend and volume confirmation
            if roc[i] < 0 and roc[i-1] >= 0:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals