#!/usr/bin/env python3
name = "1D_Aggressive_Volume_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA 21 for trend
    weekly_close = df_1w['close'].values
    ema_21_w = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_w)
    
    # Daily volume ratio
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Daily price momentum (3-day ROC)
    roc = np.zeros_like(close)
    roc[3:] = (close[3:] - close[:-3]) / close[:-3]
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_w_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(roc[i])):
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge filter
        volume_surge = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: Strong upward momentum with volume surge AND above weekly EMA
            if (roc[i] > 0.03 and 
                volume_surge and 
                close[i] > ema_21_w_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: Strong downward momentum with volume surge AND below weekly EMA
            elif (roc[i] < -0.03 and 
                  volume_surge and 
                  close[i] < ema_21_w_aligned[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: momentum reverses or volume dries up
            if position == 1:
                if (roc[i] < -0.01) or (vol_ratio[i] < 0.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                if (roc[i] > 0.01) or (vol_ratio[i] < 0.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals