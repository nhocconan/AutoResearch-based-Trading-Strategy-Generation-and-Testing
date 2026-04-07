#!/usr/bin/env python3
"""
6h_volume_weighted_price_action_v1
Hypothesis: On 6b timeframe, price rejection at institutional levels (daily VWAP ±1σ) 
combined with volume surge and higher timeframe trend (1d EMA50) provides edge.
In ranging markets: fade at VWAP bands with volume confirmation.
In trending markets: pullback to VWAP with volume confirmation.
Uses volume-weighted price action to identify institutional interest.
Target: 15-35 trades/year (60-140 over 4 years). Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_weighted_price_action_v1"
timeframe = "6h"
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
    
    # Daily data for VWAP and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP and bands
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    
    # Daily standard deviation of price from VWAP
    price_dev = typical_price_1d - vwap_1d
    variance = (price_dev ** 2).rolling(window=20, min_periods=20).mean()
    std_dev = np.sqrt(variance)
    
    # VWAP bands: VWAP ± 1 standard deviation
    vwap_upper_1d = vwap_1d + std_dev
    vwap_lower_1d = vwap_1d - std_dev
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align daily levels to 6h timeframe
    vwap_6h = align_htf_to_ltf(prices, df_1d, vwap_1d.values)
    vwap_upper_6h = align_htf_to_ltf(prices, df_1d, vwap_upper_1d.values)
    vwap_lower_6h = align_htf_to_ltf(prices, df_1d, vwap_lower_1d.values)
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average on 6h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(vwap_6h[i]) or np.isnan(vwap_upper_6h[i]) or 
            np.isnan(vwap_lower_6h[i]) or np.isnan(ema50_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below VWAP lower band OR 
            # price closes above VWAP and trend turns bearish
            if close[i] < vwap_lower_6h[i] or (close[i] > vwap_6h[i] and close[i] < ema50_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above VWAP upper band OR
            # price closes below VWAP and trend turns bullish
            if close[i] > vwap_upper_6h[i] or (close[i] < vwap_6h[i] and close[i] > ema50_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price rejects lower band with volume in uptrend
            if (close[i] <= vwap_lower_6h[i] and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price rejects upper band with volume in downtrend
            elif (close[i] >= vwap_upper_6h[i] and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
            # Long: pullback to VWAP in uptrend with volume
            elif (abs(close[i] - vwap_6h[i]) < (vwap_upper_6h[i] - vwap_6h[i]) * 0.1 and 
                  vol_confirm and 
                  close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: pullback to VWAP in downtrend with volume
            elif (abs(close[i] - vwap_6h[i]) < (vwap_upper_6h[i] - vwap_6h[i]) * 0.1 and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals