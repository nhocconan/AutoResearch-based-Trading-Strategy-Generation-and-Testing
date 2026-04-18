#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action relative to 1-day VWAP with volume confirmation.
# Long when price crosses above 1-day VWAP with volume > 1.5x 20-period average.
# Short when price crosses below 1-day VWAP with volume > 1.5x 20-period average.
# Exit when price crosses back over 1-day VWAP.
# Uses 1-day VWAP as dynamic fair value, volume surge for conviction.
# Designed for ~25-40 trades/year per symbol.
name = "4h_1dVWAP_VolumeSurge"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Typical price for VWAP
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    # VWAP calculation: cumulative sum of typical price * volume / cumulative volume
    tpv = typical_price * df_1d['volume'].values
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(df_1d['volume'].values)
    vwap = cum_tpv / cum_vol
    vwap = np.where(cum_vol > 0, vwap, typical_price)  # avoid division by zero
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vwap_val = vwap_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price crosses above VWAP with volume surge
            if close_val > vwap_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with volume surge
            elif close_val < vwap_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below VWAP
            if close_val < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above VWAP
            if close_val > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals