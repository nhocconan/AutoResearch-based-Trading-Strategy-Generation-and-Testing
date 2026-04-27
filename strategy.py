#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h VWAP-based mean reversion with 1d trend filter and volume confirmation
# Uses deviation from VWAP as mean reversion signal, filtered by 1d EMA trend
# Works in both bull/bear: buys when price deviates below VWAP in uptrend, sells when above in downtrend
# Volume filter prevents false signals. Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate VWAP for 12h data
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_numerator = (typical_price * df_12h['volume']).cumsum()
    vwap_denominator = df_12h['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap_array = vwap.values
    
    # Calculate deviation from VWAP as percentage
    deviation = (df_12h['close'].values - vwap_array) / vwap_array * 100
    
    # Align VWAP deviation to lower timeframe
    vwap_dev_aligned = align_htf_to_ltf(prices, df_12h, deviation)
    
    # Volume filter: volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap_dev_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: price below VWAP (negative deviation) in uptrend with volume
        if (vwap_dev_aligned[i] < -0.5 and  # 0.5% below VWAP
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short: price above VWAP (positive deviation) in downtrend with volume
        elif (vwap_dev_aligned[i] > 0.5 and   # 0.5% above VWAP
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold position or flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_VWAP_MeanRev_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0