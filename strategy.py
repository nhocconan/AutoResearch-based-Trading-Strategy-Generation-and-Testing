#!/usr/bin/env python3
"""
4h_trix_12h_volume_v1
Hypothesis: On 4-hour timeframe, use TRIX (1-period rate of change of triple EMA) with 12h trend filter and volume confirmation. 
TRIX crossing above zero signals momentum shift up; crossing below zero signals momentum shift down. 
Only take signals aligned with 12h EMA trend and confirmed by above-average volume. 
Designed for moderate frequency (20-50 trades/year) with clear trend signals and volatility filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX: 1-period ROC of triple EMA (12-period EMA applied 3 times)
    ema1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix = pd.Series(ema3).pct_change() * 100  # percentage change
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(25) for trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=25, adjust=False).mean()
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h.values)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 for pct_change
        # Skip if TRIX or trend data not available
        if np.isnan(trix.iloc[i]) or np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when TRIX crosses below zero OR trend turns bearish
            if trix.iloc[i] < 0 or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when TRIX crosses above zero OR trend turns bullish
            if trix.iloc[i] > 0 or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TRIX crosses above zero with volume confirmation and bullish trend
            long_entry = (trix.iloc[i-1] <= 0 and trix.iloc[i] > 0) and vol_confirm and (close[i] > ema_12h_aligned[i])
            # Short entry: TRIX crosses below zero with volume confirmation and bearish trend
            short_entry = (trix.iloc[i-1] >= 0 and trix.iloc[i] < 0) and vol_confirm and (close[i] < ema_12h_aligned[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals