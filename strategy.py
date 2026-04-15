#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-Weighted VWAP Deviation with 1d Trend Filter
# Trades when price deviates significantly from 4h VWAP (mean reversion) only when
# the 1d trend is strong (ADX > 25). Uses volume-weighted price to avoid false signals.
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 40-80 total trades to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative sums for VWAP
    cum_vwap_num = np.cumsum(vwap_numerator)
    cum_vwap_den = np.cumsum(vwap_denominator)
    vwap = np.where(cum_vwap_den > 0, cum_vwap_num / cum_vwap_den, 0)
    
    # Calculate standard deviation of price from VWAP (20-period)
    price_dev = typical_price - vwap
    # Use pandas rolling for std with min_periods
    price_dev_series = pd.Series(price_dev)
    vwap_std = price_dev_series.rolling(window=20, min_periods=20).std().values
    
    # Z-score: how many standard deviations price is from VWAP
    zscore = np.where(vwap_std > 0, price_dev / vwap_std, 0)
    
    # Calculate ADX (14-period) on 1d for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup period to ensure indicators are valid
    start_idx = max(30, 20)  # ADX needs 30, VWAP std needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or 
            np.isnan(zscore[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price significantly below VWAP (oversold) in uptrend
        if (zscore[i] < -1.5 and  # Price is 1.5+ std below VWAP
            adx_aligned[i] > 25 and  # Strong trend
            di_plus[i] > di_minus[i] and  # Plus DI > Minus DI (uptrend bias)
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price significantly above VWAP (overbought) in downtrend
        elif (zscore[i] > 1.5 and  # Price is 1.5+ std above VWAP
              adx_aligned[i] > 25 and  # Strong trend
              di_minus[i] > di_plus[i] and  # Minus DI > Plus DI (downtrend bias)
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to VWAP (mean reversion complete) or trend weakens
        elif position == 1 and (zscore[i] > -0.5 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (zscore[i] < 0.5 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_VWAP_Deviation_1dADX"
timeframe = "4h"
leverage = 1.0