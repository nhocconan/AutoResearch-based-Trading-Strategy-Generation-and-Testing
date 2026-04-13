# 1d_1w_camarilla_volume_reversal
# Hypothesis: On 1d timeframe, Camarilla pivot levels (L3/H3) from weekly data act as strong support/resistance.
# Enter reversal trades when price touches these levels with volume confirmation (volume > 1.5x 20-day average).
# Exit on opposite touch or when price moves beyond L4/H4. Works in both bull/bear as it captures mean reversion at key levels.
# Target: 20-50 trades over 4 years (5-12/year) with low turnover to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    # Using previous week's OHLC to avoid look-ahead
    camarilla_h4 = np.full(len(close_1w), np.nan)
    camarilla_h3 = np.full(len(close_1w), np.nan)
    camarilla_l3 = np.full(len(close_1w), np.nan)
    camarilla_l4 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        # Previous week's values
        high_prev = high_1w[i-1]
        low_prev = low_1w[i-1]
        close_prev = close_1w[i-1]
        range_prev = high_prev - low_prev
        
        camarilla_h4[i] = close_prev + 1.5 * range_prev
        camarilla_h3[i] = close_prev + 1.1 * range_prev
        camarilla_l3[i] = close_prev - 1.1 * range_prev
        camarilla_l4[i] = close_prev - 1.5 * range_prev
    
    # Align Camarilla levels to daily timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if Camarilla data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if volume data not ready
        if np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Check for touches of Camarilla levels (within 0.1% tolerance)
        tol = 0.001
        touch_h3 = abs(high[i] - h3_aligned[i]) / h3_aligned[i] < tol
        touch_l3 = abs(low[i] - l3_aligned[i]) / l3_aligned[i] < tol
        touch_h4 = abs(high[i] - h4_aligned[i]) / h4_aligned[i] < tol
        touch_l4 = abs(low[i] - l4_aligned[i]) / l4_aligned[i] < tol
        
        # Entry conditions: reversal at H3/L3 with volume confirmation
        long_entry = touch_l3 and volume_filter[i]
        short_entry = touch_h3 and volume_filter[i]
        
        # Exit conditions: touch opposite level (H4/L4) or reverse touch
        exit_long = position == 1 and (touch_h4 or touch_h3)
        exit_short = position == -1 and (touch_l4 or touch_l3)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_volume_reversal"
timeframe = "1d"
leverage = 1.0