#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels + 1d EMA200 trend filter + volume confirmation
# - Primary signal: Price touches Camarilla H3 (resistance) for short or L3 (support) for long
# - Trend filter: 1d EMA200 - price must be below EMA200 for shorts, above for longs (trade with higher timeframe trend)
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla pivots identify key institutional levels, EMA200 filter ensures
#   trades align with higher timeframe trend, reducing false signals in strong trends

name = "12h_1d_camarilla_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend direction
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 12h timeframe (completed 1d bar only)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    #           L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    # We need to align the previous day's Camarilla levels to current 12h bars
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First value will be invalid due to roll, but min_periods in align will handle it
    prev_range_1d = prev_high_1d - prev_low_1d
    
    camarilla_h3 = prev_close_1d + 1.1 * prev_range_1d / 4
    camarilla_l3 = prev_close_1d - 1.1 * prev_range_1d / 4
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above H3 (take profit) OR price crosses below EMA200 (trend change)
            if close[i] > h3_aligned[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below L3 (take profit) OR price crosses above EMA200 (trend change)
            if close[i] < l3_aligned[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with volume confirmation and EMA200 filter
            # Long: price touches L3 (support) AND volume regime AND price above EMA200
            if abs(close[i] - l3_aligned[i]) < (h3_aligned[i] - l3_aligned[i]) * 0.001:  # within 0.1% of L3
                if volume_regime[i] and close[i] > ema_200_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Short: price touches H3 (resistance) AND volume regime AND price below EMA200
            elif abs(close[i] - h3_aligned[i]) < (h3_aligned[i] - l3_aligned[i]) * 0.001:  # within 0.1% of H3
                if volume_regime[i] and close[i] < ema_200_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals