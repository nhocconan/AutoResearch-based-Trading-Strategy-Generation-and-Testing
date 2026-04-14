#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly Supertrend for trend filter
    atr_mult = 3.0
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (atr_mult * atr_1w)
    lower_band = hl2 - (atr_mult * atr_1w)
    
    # Initialize Supertrend arrays
    st = np.full_like(close_1w, np.nan)
    dir_ = np.full_like(close_1w, 1)  # 1 for up, -1 for down
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
            if dir_[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if dir_[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if dir_[i] == 1:
            st[i] = lower_band[i]
        else:
            st[i] = upper_band[i]
    
    st_aligned = align_htf_to_ltf(prices, df_1w, st)
    dir_aligned = align_htf_to_ltf(prices, df_1w, dir_)
    
    # Calculate weekly ATR-based volatility threshold (ATR > 1.5% of price)
    vol_threshold = 0.015
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(st_aligned[i]) or 
            np.isnan(dir_aligned[i]) or
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: Supertrend direction
        trend_filter_long = dir_aligned[i] == 1
        trend_filter_short = dir_aligned[i] == -1
        
        # Volatility filter: weekly ATR > threshold * price to avoid low volatility periods
        vol_filter = atr_1w_aligned[i] / price > vol_threshold if price > 0 else False
        
        if position == 0:
            # Long setup: Supertrend up + volatility filter
            if trend_filter_long and vol_filter:
                position = 1
                signals[i] = position_size
            # Short setup: Supertrend down + volatility filter
            elif trend_filter_short and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Supertrend flips down
            if dir_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Supertrend flips up
            if dir_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wSupertrend_ATRVol_Filter_v1"
timeframe = "12h"
leverage = 1.0