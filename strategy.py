#!/usr/bin/env python3
"""
12h_1d_VWAP_MeanReversion_Trend
Strategy: Mean-reversion from VWAP with 1d trend filter and volume confirmation.
Long: Price < VWAP by 2%, volume > 1.5x average, price above 1d EMA20
Short: Price > VWAP by 2%, volume > 1.5x average, price below 1d EMA20
Exit: Price returns to VWAP (within 0.5%)
Designed for mean-reversion in ranging markets with trend filter to avoid false signals.
Timeframe: 12h
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
    
    # Calculate VWAP
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema20_1d = close_series_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d EMA20 to 12h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or 
            np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA20
        price_above_ema = close[i] > ema20_1d_aligned[i]
        price_below_ema = close[i] < ema20_1d_aligned[i]
        
        # Mean reversion conditions
        deviation = (close[i] - vwap[i]) / vwap[i]
        oversold = deviation < -0.02  # 2% below VWAP
        overbought = deviation > 0.02  # 2% above VWAP
        
        # Return to VWAP for exit
        return_to_vwap = abs(deviation) < 0.005  # within 0.5% of VWAP
        
        if position == 0:
            # Long: oversold + volume filter + price above EMA
            if oversold and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: overbought + volume filter + price below EMA
            elif overbought and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to VWAP
            if return_to_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to VWAP
            if return_to_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_VWAP_MeanReversion_Trend"
timeframe = "12h"
leverage = 1.0