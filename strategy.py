#!/usr/bin/env python3
"""
Hypothesis: 1d price action relative to weekly VWAP with daily volume confirmation.
In bull markets: price above weekly VWAP acts as dynamic support, buy on pullbacks with volume.
In bear markets: price below weekly VWAP acts as resistance, sell on rallies with volume.
Weekly VWAP provides institutional reference point, reducing whipsaw.
Target: 15-25 trades/year to minimize fee drag on daily timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, np.nan)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate VWAP on weekly data
    vwap_1w = calculate_vwap(high_1w, low_1w, close_1w, volume_1w)
    
    # Align VWAP to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Calculate daily volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vwap_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above weekly VWAP, volume confirmation
            if close[i] > vwap_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly VWAP, volume confirmation
            elif close[i] < vwap_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly VWAP
            if close[i] <= vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly VWAP
            if close[i] >= vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_VWAP_Volume"
timeframe = "1d"
leverage = 1.0