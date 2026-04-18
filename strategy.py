#!/usr/bin/env python3
"""
12h_Stochastic_KD_Signal_With_Trend_Filter_v1
Hypothesis: Stochastic oscillator (K% and D%) crossovers on 12h timeframe provide reliable momentum signals. 
Filtered by 1d EMA50 trend direction and volume confirmation to avoid false signals in choppy markets.
Designed for low trade frequency (15-25/year) to minimize fee drag and improve generalization.
Works in bull markets via trend-following crosses and in bear markets via mean-reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Stochastic calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Stochastic K% and D% (14,3)
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * ((close_12h - lowest_low) / (highest_high - lowest_low))
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Align Stochastic to 12h timeframe (primary)
    k_aligned = align_htf_to_ltf(prices, df_12h, k_percent)
    d_aligned = align_htf_to_ltf(prices, df_12h, d_percent)
    
    # Get 1d data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: volume > 1.8 * 20-period average (12h equivalent)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(k_aligned[i]) or 
            np.isnan(d_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i]) or
            (highest_high[i] - lowest_low[i]) == 0):
            signals[i] = 0.0
            continue
        
        price = close[i]
        k = k_aligned[i]
        d = d_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: K crosses above D with uptrend and volume confirmation
            if k > d and k < 20 and ema_trend > price and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: K crosses below D with downtrend and volume confirmation
            elif k < d and k > 80 and ema_trend < price and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: K crosses below D or price breaks below EMA
            if k < d or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: K crosses above D or price breaks above EMA
            if k > d or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Stochastic_KD_Signal_With_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0