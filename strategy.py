#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h trend filter.
# Works in bull (breakouts continue) and bear (mean reversion at lower channel in range).
# Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag.
name = "4h_Donchian20_Volume_12hTrendFilter_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(34) for trend
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channel (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper = high_roll
    lower = low_roll
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        ema_val = ema_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above upper band with volume confirmation and uptrend
            if close_val > upper_val and vol_filter and (close_val > ema_val):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation and downtrend
            elif close_val < lower_val and vol_filter and (close_val < ema_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band or trend turns down
            if close_val < lower_val or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band or trend turns up
            if close_val > upper_val or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals