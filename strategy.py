#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(15) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for trend bias, reducing false signals in chop and bear markets
# Donchian(15) channels identify key support/resistance from 6h price action
# Breakout above upper band or below lower band with volume confirmation signals momentum
# EMA50 filter ensures trades align with higher timeframe trend
# Target: 15-25 trades/year per symbol with disciplined entries
name = "6h_Donchian15_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (15-period) from 6h data
    # Upper band = highest high of last 15 periods
    # Lower band = lowest low of last 15 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=15, min_periods=15).max().values
    lower_band = low_series.rolling(window=15, min_periods=15).min().values
    
    # Volume confirmation: volume > 1.5 * 15-period average
    volume_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volume confirmation and above 1d EMA50
            if (close[i] > upper_band[i] and 
                volume_confirm[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation and below 1d EMA50
            elif (close[i] < lower_band[i] and 
                  volume_confirm[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower band or breaks below 1d EMA50
            if (close[i] < lower_band[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper band or breaks above 1d EMA50
            if (close[i] > upper_band[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals