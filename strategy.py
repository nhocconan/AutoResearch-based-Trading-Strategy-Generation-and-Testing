#!/usr/bin/env python3
"""
12h_1w_Volume_Weighted_VWAP_Slope_Trend
Hypothesis: Buy when VWAP slope on 1w is positive and volume is above average, with price above VWAP; short when slope negative, volume above average, price below VWAP. Uses weekly VWAP slope as trend filter and volume confirmation to avoid false signals. Designed for low trade frequency (<30/year) to minimize fee decay while capturing sustained trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-week data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate VWAP on 1w: cumulative (price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap = vwap.values
    
    # VWAP slope: 5-period linear regression slope (approximate with 5-bar difference)
    vwap_slope = np.zeros_like(vwap)
    for i in range(5, len(vwap)):
        vwap_slope[i] = (vwap[i] - vwap[i-5]) / 5
    
    # Volume average on 1w (20-period)
    vol_ma_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align VWAP, slope, and volume MA to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    vwap_slope_aligned = align_htf_to_ltf(prices, df_1w, vwap_slope)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need sufficient history for VWAP and slope
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(vwap_slope_aligned[i]) or
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap_aligned[i]
        slope = vwap_slope_aligned[i]
        vol_ma = vol_ma_1w_aligned[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x weekly average volume
        vol_confirm = vol > (1.5 * vol_ma)
        
        if position == 0:
            # Long: price above VWAP, positive slope, volume confirmation
            if price > vwap_val and slope > 0 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP, negative slope, volume confirmation
            elif price < vwap_val and slope < 0 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below VWAP or slope turns negative
            if price < vwap_val or slope < 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above VWAP or slope turns positive
            if price > vwap_val or slope > 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1w_Volume_Weighted_VWAP_Slope_Trend"
timeframe = "12h"
leverage = 1.0