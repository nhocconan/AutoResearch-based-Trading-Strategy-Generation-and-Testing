#!/usr/bin/env python3
"""
6H_Chaikin_Oscillator_CCI_Breakout_1dTrend_v1
Hypothesis: Use 1d CCI(20) for trend direction and 6h Chaikin Oscillator for momentum confirmation.
Long when CCI > 100 (uptrend) and Chaikin Oscillator crosses above zero with volume confirmation.
Short when CCI < -100 (downtrend) and Chaikin Oscillator crosses below zero.
Add volume filter: current volume > 1.3x 20-period average volume.
This combines trend-following with volume-weighted momentum to reduce false signals in both bull and bear markets.
Target: 12-37 trades/year on 6h timeframe.
"""
name = "6H_Chaikin_Oscillator_CCI_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for CCI trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d CCI(20)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp = typical_price_1d.rolling(window=20, min_periods=20).mean()
    mean_deviation = typical_price_1d.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price_1d - sma_tp) / (0.015 * mean_deviation.replace(0, np.nan))
    cci_values = cci.values
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_values)
    
    # Calculate 6h Chaikin Oscillator (3,10)
    # ADL = cumulative sum of ((close - low) - (high - close)) / (high - low) * volume
    adl_raw = ((close - low) - (high - close)) / (high - low)
    adl_raw = np.where((high - low) == 0, 0, adl_raw)  # avoid division by zero
    adl = np.cumsum(adl_raw * volume)
    
    # Chaikin Oscillator = EMA(3) of ADL - EMA(10) of ADL
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean()
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean()
    chaikin_osc = (ema3 - ema10).values
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(cci_aligned[i]) or np.isnan(chaikin_osc[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades (1.5 days on 6h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Long: CCI > 100 (uptrend) and Chaikin Oscillator crosses above zero
            if (cci_aligned[i] > 100 and 
                chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0 and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: CCI < -100 (downtrend) and Chaikin Oscillator crosses below zero
            elif (cci_aligned[i] < -100 and 
                  chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0 and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: Chaikin Oscillator crosses zero in opposite direction
            if position == 1 and chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals