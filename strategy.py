#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index with 1-week EMA20 trend filter and volume confirmation
# Choppiness Index > 61.8 indicates ranging market (mean reversion opportunity)
# Choppiness Index < 38.2 indicates trending market (trend following)
# Combined with 1-week EMA20 to filter trades in direction of higher timeframe trend
# Volume confirmation ensures institutional participation
# Target: 30-100 total trades over 4 years (7-25/year)
name = "1d_Choppiness_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Choppiness Index (14-period)
    atr_list = []
    for i in range(n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]) if i > 0 else 0, abs(low[i] - close[i-1]) if i > 0 else 0)
        atr_list.append(tr)
    atr = np.array(atr_list)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.where(range_hl != 0, -100 * np.log10(atr_sum / range_hl) / np.log10(14), 50)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # In ranging market (CHOP > 61.8): mean reversion at extremes
            if (chop[i] > 61.8 and 
                close[i] < ll[i] and  # Near low of range - look for bounce
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            elif (chop[i] > 61.8 and 
                  close[i] > hh[i] and  # Near high of range - look for reversal
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            # In trending market (CHOP < 38.2): follow trend with 1-week EMA
            elif (chop[i] < 38.2 and 
                  close[i] > ema_20_1w_aligned[i] and  # Uptrend
                  volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            elif (chop[i] < 38.2 and 
                  close[i] < ema_20_1w_aligned[i] and  # Downtrend
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: chop shifts to trending down OR price crosses below weekly EMA
            if (chop[i] < 38.2 and close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: chop shifts to trending up OR price crosses above weekly EMA
            if (chop[i] < 38.2 and close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals