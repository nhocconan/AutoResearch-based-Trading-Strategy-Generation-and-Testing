#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot with 1d Volume and Trend Filter
# Hypothesis: Camarilla pivot levels from 1d provide strong support/resistance.
# In uptrend (price > 1d EMA50), buy at L3 level with volume confirmation.
# In downtrend (price < 1d EMA50), sell at H3 level with volume confirmation.
# Uses volume spike (>1.5x average) to confirm institutional interest.
# Target: 20-40 trades/year (80-160 over 4 years) to avoid fee drag.

name = "4h_camarilla_pivot_1d_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: H4, H3, L3, L4
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    camarilla_h3 = close_1d_prev + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d_prev - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 1.5x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below L3 or trend turns down
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_1d_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price above H3 or trend turns up
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_1d_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation
            vol_confirm = volume[i] > 1.5 * volume_ma[i]
            
            # Long: uptrend + price at L3 support + volume
            if (close[i] > ema_1d_50_aligned[i] and 
                abs(camarilla_l3_aligned[i] - close[i]) / close[i] < 0.005 and  # Within 0.5% of L3
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: downtrend + price at H3 resistance + volume
            elif (close[i] < ema_1d_50_aligned[i] and 
                  abs(camarilla_h3_aligned[i] - close[i]) / close[i] < 0.005 and  # Within 0.5% of H3
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals