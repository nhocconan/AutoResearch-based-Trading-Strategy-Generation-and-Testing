#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 12-hour Volume Weighted Average Price (VWAP) with
# daily Bollinger Band mean reversion. VWAP acts as dynamic support/resistance,
# while Bollinger Bands identify overbought/oversold conditions. Long when price
# pulls back to VWAP from below during oversold conditions (BB < 20), short when
# price rallies to VWAP from above during overbought conditions (BB > 80).
# This mean-reversion-to-VWAP approach works in both trending and ranging markets
# by fading extremes relative to the dynamic VWAP mean. Volume-weighted pricing
# reduces false signals from low-volume spikes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate VWAP on 12h data: cumulative(volume * price) / cumulative(volume)
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    vp_12h = typical_price_12h * df_12h['volume'].values
    cum_vp_12h = np.cumsum(vp_12h)
    cum_vol_12h = np.cumsum(df_12h['volume'].values)
    vwap_12h = cum_vp_12h / cum_vol_12h
    
    # Align 12h VWAP to 6h timeframe (wait for 12h bar close)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Load daily data ONCE for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on daily close
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    
    # Calculate %B: (close - lower) / (upper - lower)
    bb_range = bb_upper - bb_lower
    # Avoid division by zero
    bb_range_safe = np.where(bb_range == 0, 1e-10, bb_range)
    percent_b = (close_1d - bb_lower) / bb_range_safe
    
    # Align Bollinger %B to 6h timeframe
    percent_b_aligned = align_htf_to_ltf(prices, df_1d, percent_b)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # Need Bollinger Bands
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_12h_aligned[i]) or 
            np.isnan(percent_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for mean reversion to VWAP from Bollinger extremes
            # Long: price below VWAP AND oversold (%B < 0.2)
            if (close[i] < vwap_12h_aligned[i] and 
                percent_b_aligned[i] < 0.2):
                position = 1
                signals[i] = position_size
            # Short: price above VWAP AND overbought (%B > 0.8)
            elif (close[i] > vwap_12h_aligned[i] and 
                  percent_b_aligned[i] > 0.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches VWAP or becomes overbought
            if (close[i] >= vwap_12h_aligned[i] or 
                percent_b_aligned[i] > 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches VWAP or becomes oversold
            if (close[i] <= vwap_12h_aligned[i] or 
                percent_b_aligned[i] < 0.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12hVWAP_1dBB_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0