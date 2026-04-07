#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Mean Reversion with 1d Range and Volume Confirmation
# Hypothesis: Price reverts to the mean of the previous day's range after extreme moves.
# In both bull and bear markets, price tends to revert to the daily value area (VAH/VAL).
# Uses 1d value area (VWAP-based) as mean, with volume confirmation and time filter.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "1h_mean_reversion_1d_vwap_volume_v1"
timeframe = "1h"
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
    
    # Get daily data for VWAP calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP (Volume Weighted Average Price)
    typical_price = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 3
    vwap = (typical_price * df_daily['volume']).cumsum() / df_daily['volume'].cumsum()
    vwap_values = vwap.values
    
    # Use previous day's VWAP (avoid look-ahead)
    prev_vwap = np.roll(vwap_values, 1)
    prev_vwap[0] = prev_vwap[1] if len(prev_vwap) > 1 else vwap_values[0]
    
    # Align to 1h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_daily, prev_vwap)
    
    # Volume filter: volume > 1.2x 24-period average (to avoid low-volume noise)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.2 * vol_ma)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available or outside session
        if (np.isnan(vwap_aligned[i]) or np.isnan(vol_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP or volume drops
            if (close[i] >= vwap_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price returns to VWAP or volume drops
            if (close[i] <= vwap_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long: price is significantly below VWAP with volume confirmation
            if (close[i] < (vwap_aligned[i] * 0.995) and vol_filter[i]):  # 0.5% below VWAP
                position = 1
                signals[i] = 0.20
            # Short: price is significantly above VWAP with volume confirmation
            elif (close[i] > (vwap_aligned[i] * 1.005) and vol_filter[i]):  # 0.5% above VWAP
                position = -1
                signals[i] = -0.20
    
    return signals