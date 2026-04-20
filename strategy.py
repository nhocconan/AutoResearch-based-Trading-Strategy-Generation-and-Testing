#!/usr/bin/env python3
"""
6h_1d_Volume_Weighted_Trend_Acceleration_v1
Concept: 6h acceleration of volume-weighted trend with 1d volatility filter.
- Long: 6h VWAP trending up + 6h price acceleration + 1d volatility low (avoid whipsaw)
- Short: 6h VWAP trending down + 6h price deceleration + 1d volatility low
- Exit: VWAP trend reversal or volatility spike
- Uses volume-weighted price to reduce noise, acceleration for momentum, volatility filter for regime
- Works in bull/bear: volatility filter adapts to market conditions, VWAP trend captures institutional flow
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Volume_Weighted_Trend_Acceleration_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 6h: Volume Weighted Average Price (VWAP) ===
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3.0
    vwap_numerator = (typical_price * prices['volume']).cumsum()
    vwap_denominator = prices['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap = vwap.values  # Convert to numpy array
    
    # === 6h: Price acceleration (2nd derivative of price) ===
    close = prices['close'].values
    # First derivative (velocity): price change over 3 periods
    velocity = np.zeros(n)
    velocity[3:] = close[3:] - close[:-3]
    # Second derivative (acceleration): change in velocity
    acceleration = np.zeros(n)
    acceleration[6:] = velocity[6:] - velocity[:-6]
    
    # === Daily: Volatility filter (low volatility = better for trend following) ===
    # Use ATR-based volatility normalized by price
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # ATR(10)
    atr_10 = np.zeros(len(tr))
    for i in range(10, len(tr)):
        atr_10[i] = np.nanmean(tr[i-9:i+1])
    
    # Volatility ratio: ATR(10) / price
    vol_ratio = atr_10 / close_1d
    vol_ratio = np.where(close_1d == 0, np.nan, vol_ratio)
    
    # Volatility percentile (low volatility = good)
    vol_percentile = np.zeros_like(vol_ratio)
    for i in range(len(vol_ratio)):
        if i < 20:
            vol_percentile[i] = np.nan
        else:
            vol_percentile[i] = np.nanpercentile(vol_ratio[max(0, i-19):i+1], 50)
    
    # Align volatility percentile to 6h timeframe
    vol_percentile_aligned = align_htf_to_ltf(prices, df_1d, vol_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        vwap_val = vwap[i]
        price = close[i]
        accel = acceleration[i]
        vol_percentile_val = vol_percentile_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vwap_val) or np.isnan(price) or np.isnan(accel) or 
            np.isnan(vol_percentile_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above VWAP AND positive acceleration AND low volatility
            if price > vwap_val and accel > 0 and vol_percentile_val < 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Price below VWAP AND negative acceleration AND low volatility
            elif price < vwap_val and accel < 0 and vol_percentile_val < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below VWAP OR volatility spikes
            if price < vwap_val or vol_percentile_val > 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above VWAP OR volatility spikes
            if price > vwap_val or vol_percentile_val > 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals