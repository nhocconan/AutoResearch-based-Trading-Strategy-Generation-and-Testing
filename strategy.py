#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + 12h EMA trend + volume confirmation
# - Primary signal: Price touches Camarilla H3/L3 levels from 1d timeframe
# - Trend filter: 12h EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 4h volume > 20-period median volume
# - Exit: Price touches opposite Camarilla level (H4/L4) or EMA cross
# - Position size: 0.25 (discrete level)
# - Works in bull/bear: Camarilla provides mean-reversion levels, EMA filter ensures trend alignment
# - Target: 20-50 trades/year (80-200 total over 4 years)

name = "4h_1d_12h_camarilla_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, L3, L4
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to primary timeframe (completed 1d bar only)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(h4_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or
            np.isnan(l4_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches H4 OR price crosses below 12h EMA50
            if high[i] >= h4_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches L4 OR price crosses above 12h EMA50
            if low[i] <= l4_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla touch with volume confirmation and 12h EMA50 filter
            # Long: price touches H3 from below AND volume regime AND price above 12h EMA50
            if (low[i] <= h3_aligned[i] and 
                close[i] > h3_aligned[i] and  # confirmation close above level
                volume_regime[i] and 
                close[i] > ema_50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches L3 from above AND volume regime AND price below 12h EMA50
            elif (high[i] >= l3_aligned[i] and 
                  close[i] < l3_aligned[i] and  # confirmation close below level
                  volume_regime[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals