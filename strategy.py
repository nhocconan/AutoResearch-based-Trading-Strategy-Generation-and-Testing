#!/usr/bin/env python3
"""
12h Camarilla Pivot with 1w Trend Filter and Volume Confirmation
Hypothesis: Camarilla pivot levels on 12h timeframe provide high-probability reversal points when aligned with 1w trend and volume spikes, offering robust performance in both bull and bear markets by capturing mean-reversion within the dominant trend. Designed for 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily data for Camarilla pivots (calculate from previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 12h bar using prior day's data
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # We'll use these as key reversal levels
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume filter: current volume > 2.0x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches Camarilla resistance OR trend reverses
            if (close[i] >= camarilla_high_aligned[i] or 
                close[i] < ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla support OR trend reverses
            if (close[i] <= camarilla_low_aligned[i] or 
                close[i] > ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry at Camarilla levels with rejection
            # Look for price rejection at Camarilla levels with volume spike
            near_high = abs(high[i] - camarilla_high_aligned[i]) < (high[i] * 0.002)  # Within 0.2%
            near_low = abs(low[i] - camarilla_low_aligned[i]) < (low[i] * 0.002)      # Within 0.2%
            
            # Long: rejection at support (low touches L4 then closes above) in uptrend
            if (near_low and 
                close[i] > camarilla_low_aligned[i] and 
                close[i] > open_prices[i] and  # Bullish close
                close[i] > ema_20_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: rejection at resistance (high touches H4 then closes below) in downtrend
            elif (near_high and 
                  close[i] < camarilla_high_aligned[i] and 
                  close[i] < open_prices[i] and  # Bearish close
                  close[i] < ema_20_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals

# Fix: need to access open prices
    open_prices = prices['open'].values
    # Replace open_prices[i] in the loop with the actual open prices array
    
    # Rewriting the function with proper open price access:
    
    # ... (previous code remains the same until the loop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches Camarilla resistance OR trend reverses
            if (close[i] >= camarilla_high_aligned[i] or 
                close[i] < ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla support OR trend reverses
            if (close[i] <= camarilla_low_aligned[i] or 
                close[i] > ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry at Camarilla levels with rejection
            # Look for price rejection at Camarilla levels with volume spike
            near_high = abs(high[i] - camarilla_high_aligned[i]) < (high[i] * 0.002)  # Within 0.2%
            near_low = abs(low[i] - camarilla_low_aligned[i]) < (low[i] * 0.002)      # Within 0.2%
            
            # Long: rejection at support (low touches L4 then closes above) in uptrend
            if (near_low and 
                close[i] > camarilla_low_aligned[i] and 
                close[i] > open_prices[i] and  # Bullish close
                close[i] > ema_20_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: rejection at resistance (high touches H4 then closes below) in downtrend
            elif (near_high and 
                  close[i] < camarilla_high_aligned[i] and 
                  close[i] < open_prices[i] and  # Bearish close
                  close[i] < ema_20_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals

# Final correction: need to define open_prices before using it
    
    # Price data
    open_prices = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Rest remains the same...