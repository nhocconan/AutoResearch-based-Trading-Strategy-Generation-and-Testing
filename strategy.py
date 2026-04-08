#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_breakout_v1
# Hypothesis: Price breaking Camarilla pivot levels (H4/L4) on 12h timeframe with volume confirmation and 1-day trend filter.
# Long when price breaks above H4 (1.125) with volume > 1.5x 20-period average and 1-day close > 1-day SMA(50).
# Short when price breaks below L4 (0.875) with volume > 1.5x 20-period average and 1-day close < 1-day SMA(50).
# Uses 12h timeframe for entries and 1-day for trend filter to reduce whipsaw. Designed for 15-35 trades/year.
# Works in bull markets via upside breakouts and bear markets via downside breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h SMA(50) for trend filter
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # 12h volume MA(20) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from previous day's range
    # H4 = close + 1.5 * (high - low) * 1.125
    # L4 = close - 1.5 * (high - low) * 1.125
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First value will be incorrect due to roll, but will be handled by min_periods
    
    # Calculate Camarilla levels
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low) * 1.125
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low) * 1.125
    
    # Align Camarilla levels to 12h timeframe (already delayed by 1 day due to roll)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure SMA(50) and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(sma50[i]) or np.isnan(vol_ma_20[i]) or np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # 1-day trend filter: close > SMA(50) for long, close < SMA(50) for short
        # Get 1-day SMA(50) aligned to 12h timeframe
        sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
        sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
        
        if np.isnan(sma50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
            
        if position == 1:  # Long position
            # Exit: price breaks below L4 (0.875 level) or trend reverses
            if close[i] < camarilla_l4_aligned[i] or close[i] < sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above H4 (1.125 level) or trend reverses
            if close[i] > camarilla_h4_aligned[i] or close[i] > sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above H4 with volume surge and uptrend
            if close[i] > camarilla_h4_aligned[i] and vol_surge and close[i] > sma50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume surge and downtrend
            elif close[i] < camarilla_l4_aligned[i] and vol_surge and close[i] < sma50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals