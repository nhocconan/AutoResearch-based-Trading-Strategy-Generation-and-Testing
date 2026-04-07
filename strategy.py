#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: On 4h timeframe, use daily Camarilla pivot levels with volume confirmation. 
Enter long when price breaks above H3 with volume > 1.5x average; enter short when price breaks below L3 with volume > 1.5x average. 
Exit on opposite signal or when price returns to Pivot level. Works in bull/bear via mean reversion at extreme levels. 
Targets 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
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
    
    # Calculate ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Camarilla pivots (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.0*(High-Low), etc.
    # L3 = Close - 1.0*(High-Low), L4 = Close - 1.5*(High-Low)
    # Pivot = (High + Low + Close)/3
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + 1.0 * range_val
    L3 = pivot - 1.0 * range_val
    H4 = pivot + 1.5 * range_val
    L4 = pivot - 1.5 * range_val
    
    # Align to 4h timeframe (shifted by 1 day to avoid look-ahead)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or atr[i] <= 0 or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on short signal (price breaks below L3 with volume)
            if close[i] < L3_aligned[i] and vol_confirm:
                exit_long = True
            # Exit when price returns to pivot level (mean reversion)
            elif abs(close[i] - pivot_aligned[i]) < 0.5 * atr[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on long signal (price breaks above H3 with volume)
            if close[i] > H3_aligned[i] and vol_confirm:
                exit_short = True
            # Exit when price returns to pivot level (mean reversion)
            elif abs(close[i] - pivot_aligned[i]) < 0.5 * atr[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above H3 with volume confirmation
            long_entry = close[i] > H3_aligned[i] and vol_confirm
            
            # Short entry: price breaks below L3 with volume confirmation
            short_entry = close[i] < L3_aligned[i] and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals