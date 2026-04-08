#!/usr/bin/env python3
# 12h_1d_williams_alligator_v1
# Hypothesis: Williams Alligator (13,8,5 SMAs) on 1-day timeframe with volume confirmation and ATR-based volatility filter.
# Long when 13-day SMA > 8-day SMA > 5-day SMA (bullish alignment) with volume > 1.3x 20-period average and ATR(14) > median ATR(100).
# Short when 13-day SMA < 8-day SMA < 5-day SMA (bearish alignment) with same volume and volatility filters.
# Uses 12-hour timeframe for entries to capture intra-day moves while using daily trend filter to avoid whipsaws.
# Designed for 12-37 trades/year on 12H timeframe to minimize fee decay while capturing sustained trends.
# Works in bull markets via sustained uptrends and bear markets via sustained downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range and ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR(100) for volatility regime filter
    atr100 = pd.Series(tr).ewm(span=100, adjust=False, min_periods=100).mean().values
    median_atr100 = pd.Series(atr100).rolling(window=100, min_periods=100).median().values
    
    # Volatility filter: current ATR > median ATR
    vol_filter = atr14 > median_atr100
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day OHLC data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs with specific periods
    # Jaw (13-period smoothed, 8-period ahead)
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw_1d = jaw_1d.shift(8)  # 8-period ahead shift
    
    # Teeth (8-period smoothed, 5-period ahead)
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth_1d = teeth_1d.shift(5)  # 5-period ahead shift
    
    # Lips (5-period smoothed, 3-period ahead)
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips_1d = lips_1d.shift(3)  # 3-period ahead shift
    
    # Align Alligator lines to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d.values)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d.values)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure ATR(100) and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vol_ma_20[i]) or np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or np.isnan(vol_filter[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.3 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: bearish alignment (jaws < teeth < lips)
            if jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish alignment (jaws > teeth > lips)
            if jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish alignment (jaws > teeth > lips) with volume surge and volatility filter
            if jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i] and vol_surge and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment (jaws < teeth < lips) with volume surge and volatility filter
            elif jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i] and vol_surge and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals