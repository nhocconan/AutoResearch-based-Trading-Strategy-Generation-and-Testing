#!/usr/bin/env python3
# 1d_camarilla_1w_trend_volume_v1
# Hypothesis: Daily Camarilla pivot levels from 1w HTF for trend context, combined with 1d price touching Camarilla levels + volume confirmation.
# In uptrend (price above 1w Camarilla H3), look for longs at L3/L4 support with volume spike.
# In downtrend (price below 1w Camarilla L3), look for shorts at H3/H4 resistance with volume spike.
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-120 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_1w_trend_volume_v1"
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
    
    # 1w HTF data for Camarilla pivot levels (trend context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels (based on previous bar's range)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    rng = high_1w - low_1w
    h4 = close_1w + 1.5 * rng
    h3 = close_1w + 1.1 * rng
    l3 = close_1w - 1.1 * rng
    l4 = close_1w - 1.5 * rng
    
    # Align Camarilla levels to 1d timeframe (completed 1w bar only)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L4 (stop loss) OR reaches H3 (take profit)
            if close[i] < l4_aligned[i] or close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H4 (stop loss) OR reaches L3 (take profit)
            if close[i] > h4_aligned[i] or close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Determine trend context from 1w Camarilla
                # Uptrend: price above weekly H3
                # Downtrend: price below weekly L3
                if close[i] > h3_aligned[i]:  # Uptrend context
                    # Long at L3/L4 support with volume spike
                    if close[i] <= l3_aligned[i] * 1.001 or close[i] <= l4_aligned[i] * 1.001:
                        position = 1
                        signals[i] = 0.25
                elif close[i] < l3_aligned[i]:  # Downtrend context
                    # Short at H3/H4 resistance with volume spike
                    if close[i] >= h3_aligned[i] * 0.999 or close[i] >= h4_aligned[i] * 0.999:
                        position = -1
                        signals[i] = -0.25
    
    return signals