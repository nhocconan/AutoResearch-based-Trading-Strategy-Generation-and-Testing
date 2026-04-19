#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price closes outside 1-day Bollinger Bands(20,2) with volume confirmation
# - Bollinger Bands calculated on 1-day close prices
# - Long when price closes below lower BBAND and 6h volume > 1.5x 20-period average 6h volume
# - Short when price closes above upper BBAND and 6h volume > 1.5x 20-period average 6h volume
# - Exit when price returns inside the Bollinger Bands
# - Designed to capture mean reversion after volatility spikes in both bull and bear markets
# - Target: 15-30 trades/year to minimize fee drag

name = "6h_BollingerBand_1dReversion_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Bollinger Bands on 1d close: 20-period SMA ± 2*stddev
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # 20-period average volume on 6x timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average volume
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: price below lower Bollinger Band + volume spike
            if close[i] < lower_bb_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price above upper Bollinger Band + volume spike
            elif close[i] > upper_bb_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns inside Bollinger Bands
            if close[i] >= lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns inside Bollinger Bands
            if close[i] <= upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals