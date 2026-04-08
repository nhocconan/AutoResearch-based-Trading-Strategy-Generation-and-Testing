#!/usr/bin/env python3
# 1d_1w_volatility_breakout_v1
# Hypothesis: Daily volatility breakout above 1-week high/low with volume confirmation and volatility filter.
# Long when daily close > 1-week high with volume > 1.5x 20-day average and ATR(14) > median ATR(100).
# Short when daily close < 1-week low with volume > 1.5x 20-day average and ATR(14) > median ATR(100).
# Designed for 10-30 trades/year on daily timeframe to minimize fee decay while capturing volatility expansion.
# Works in bull markets via upside breakouts and bear markets via downside breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_volatility_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Median ATR(100) for volatility regime filter
    atr100 = pd.Series(tr).ewm(span=100, adjust=False, min_periods=100).mean().values
    median_atr100 = pd.Series(atr100).rolling(window=100, min_periods=100).median().values
    
    # Volatility filter: current ATR > median ATR
    vol_filter = atr14 > median_atr100
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-week high/low data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align 1-week high/low to daily timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure ATR(100) and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vol_ma_20[i]) or np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or np.isnan(vol_filter[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: close below 1-week low
            if close[i] < low_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above 1-week high
            if close[i] > high_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: close above 1-week high with volume surge and volatility filter
            if close[i] > high_1w_aligned[i] and vol_surge and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: close below 1-week low with volume surge and volatility filter
            elif close[i] < low_1w_aligned[i] and vol_surge and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals