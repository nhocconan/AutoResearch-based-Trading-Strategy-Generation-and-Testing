#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Bollinger Bands mean reversion with volume spike confirmation.
# Long when price touches lower Bollinger Band (20,2) + volume > 2.0x 20-period median volume.
# Short when price touches upper Bollinger Band (20,2) + volume > 2.0x 20-period median volume.
# Exit when price returns to Bollinger middle (20-period SMA) or when volume drops below median.
# Uses discrete position size 0.25. Bollinger Bands capture volatility and mean reversion in ranging markets.
# Volume spike ensures institutional participation at extremes. 12h timeframe targets 15-35 trades/year to minimize fee drag.
# Works in both bull and bear markets by fading extremes with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Bollinger Bands and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # === 1d Indicators: Bollinger Bands (20,2) ===
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2.0 * std_20)
    bb_lower = sma_20 - (2.0 * std_20)
    bb_middle = sma_20
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (12h)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # Bollinger Bands and volume median need 20 periods
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or np.isnan(bb_middle_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        middle = bb_middle_aligned[i]
        vol_median = vol_median_aligned[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 2.0x median volume
        volume_spike = current_vol_1d > (vol_median * 2.0)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle OR volume drops below median
            if (price >= middle) or (current_vol_1d < vol_median):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle OR volume drops below median
            if (price <= middle) or (current_vol_1d < vol_median):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price touches lower Bollinger Band + volume spike
            if (price <= lower) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price touches upper Bollinger Band + volume spike
            elif (price >= upper) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dBollingerBands20_2_VolumeSpike2.0x_MeanReversion_V1"
timeframe = "12h"
leverage = 1.0