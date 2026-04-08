#!/usr/bin/env python3
# 1h_4h_1d_roc_volume_trend_follow_v1
# Hypothesis: Trend following using ROC momentum with volume confirmation and 4h/1d trend filters.
# In bull markets: long when 1h ROC > 0 with volume surge and 4h/1d uptrend.
# In bear markets: short when 1h ROC < 0 with volume surge and 4h/1d downtrend.
# Uses ROC for momentum strength, volume for conviction, and higher timeframes for direction.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_roc_volume_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h ROC(12) for momentum
    roc = np.zeros_like(close)
    roc[12:] = (close[12:] - close[:-12]) / close[:-12] * 100
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h EMA21 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(roc[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: ROC turns negative or trend breaks (price < 4h EMA21)
            if roc[i] < 0 or close[i] < ema21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: ROC turns positive or trend breaks (price > 4h EMA21)
            if roc[i] > 0 or close[i] > ema21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: ROC > 0 with volume surge and 4h/1d uptrend
            if (roc[i] > 0 and vol_surge and 
                close[i] > ema21_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: ROC < 0 with volume surge and 4h/1d downtrend
            elif (roc[i] < 0 and vol_surge and 
                  close[i] < ema21_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals