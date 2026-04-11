#!/usr/bin/env python3
"""
4h_1d_Volume_Weighted_Price_Action
Hypothesis: Uses daily VWAP deviation + volume imbalance to detect institutional accumulation/distribution.
Trades when price deviates significantly from daily VWAP with confirming volume flow.
Works in bull/bear by capturing mean-reversion moves during high-volume institutional activity.
Designed for 20-40 trades/year per symbol with focus on high-probability setups.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Volume_Weighted_Price_Action"
timeframe = "4h"
leverage = 1.0

def calculate_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Calculate Volume Weighted Average Price for given arrays."""
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d)
    
    # Calculate price deviation from VWAP as percentage
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_deviation = (typical_price_1d - vwap_1d) / vwap_1d * 100  # percentage deviation
    
    # Calculate volume imbalance (buying vs selling pressure)
    # Using close location within day's range weighted by volume
    price_range = high_1d - low_1d
    # Avoid division by zero
    price_range = np.where(price_range == 0, 1, price_range)
    close_location = (close_1d - low_1d) / price_range  # 0=low, 0.5=mid, 1=high
    volume_imbalance = (2 * close_location - 1) * volume_1d  # -1 to 1 scaled by volume
    
    # Smooth volume imbalance to reduce noise
    volume_imbalance_smooth = pd.Series(volume_imbalance).ewm(span=10, adjust=False).values
    
    # Align all to 4h timeframe
    vwap_deviation_aligned = align_htf_to_ltf(prices, df_1d, vwap_deviation)
    volume_imbalance_aligned = align_htf_to_ltf(prices, df_1d, volume_imbalance_smooth)
    
    # 4h volume filter: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_deviation_aligned[i]) or 
            np.isnan(volume_imbalance_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Mean reversion signals from VWAP deviation
        # Long when price significantly below VWAP with buying pressure
        long_signal = (vwap_deviation_aligned[i] < -1.5) and (volume_imbalance_aligned[i] > 0) and volume_filter
        
        # Short when price significantly above VWAP with selling pressure
        short_signal = (vwap_deviation_aligned[i] > 1.5) and (volume_imbalance_aligned[i] < 0) and volume_filter
        
        # Exit when price returns to VWAP or volume dries up
        vwap_return = abs(vwap_deviation_aligned[i]) < 0.5
        low_volume = volume[i] < 0.5 * vol_ma_20[i]
        exit_signal = vwap_return or low_volume
        
        # Priority: entry > exit > hold
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_signal:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_signal:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals