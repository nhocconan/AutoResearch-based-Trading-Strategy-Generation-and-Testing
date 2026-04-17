#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w structure and 1d momentum filter.
Trade weekly Donchian channel breakouts (20-period) with daily RSI momentum filter and volume confirmation.
Use daily timeframe to reduce trade frequency (target 10-25/year) and avoid fee drag.
Works in bull markets via trend-following breakouts and in bear via mean-reversion at weekly structure.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for structure (Donchian channels)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channels (20-period)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14) for momentum filter
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    
    # Align 1w and 1d data to 1d
    high_max_20_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # Volume filter: current volume > 1.5x 20-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian high with volume and bullish momentum (RSI > 50)
            if close[i] > high_max_20_aligned[i] and volume_filter[i] and rsi_14_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian low with volume and bearish momentum (RSI < 50)
            elif close[i] < low_min_20_aligned[i] and volume_filter[i] and rsi_14_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 1w Donchian low (mean reversion)
            if close[i] < low_min_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 1w Donchian high (mean reversion)
            if close[i] > high_max_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wDonchian20_1dRSI50_Volume"
timeframe = "1d"
leverage = 1.0