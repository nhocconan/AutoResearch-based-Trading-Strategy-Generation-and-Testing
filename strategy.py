#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Volume Weighted Average Price (VWAP) deviation + volume surge + volatility filter.
# VWAP deviation identifies mean reversion opportunities when price deviates significantly from VWAP.
# Volume surge confirms institutional interest, volatility filter ensures sufficient market movement.
# Designed for low trade frequency (<30/year) to minimize fee decay while capturing mean reversion in ranging markets.
# Works in both bull/bear markets by fading extreme deviations from institutional VWAP.

name = "12h_1d_vwap_deviation_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d VWAP (Volume Weighted Average Price)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price_1d * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap_1d = (vwap_numerator / vwap_denominator).values
    # Handle division by zero on first bar
    vwap_1d[0] = typical_price_1d.iloc[0] if hasattr(typical_price_1d, 'iloc') else typical_price_1d[0]
    
    # Calculate 1d standard deviation of price from VWAP (20-period)
    price_dev_1d = typical_price_1d - vwap_1d
    vwap_std_20_1d = pd.Series(price_dev_1d).rolling(window=20, min_periods=20).std().values
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align VWAP, its std dev, and volume average to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vwap_std_aligned = align_htf_to_ltf(prices, df_1d, vwap_std_20_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 12h typical price for deviation measurement
    typical_price = (high + low + close) / 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure VWAP std and volume averages are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_aligned[i]) or np.isnan(vwap_std_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: current VWAP std > 1.5 * median of last 30 periods
        vol_filter = vwap_std_aligned[i] > 1.5 * np.nanmedian(vwap_std_aligned[max(0, i-30):i])
        
        # Volume filter: current volume > 2.0 * 1d average volume (higher threshold for fewer trades)
        vol_surge = volume[i] > 2.0 * vol_avg_aligned[i]
        
        # Deviation from VWAP: price deviation in units of VWAP std dev
        price_deviation = (typical_price[i] - vwap_aligned[i]) / vwap_std_aligned[i]
        
        # Entry conditions: price deviates >2.0 std dev from VWAP with volatility and volume surge
        # Long when price is significantly below VWAP (mean reversion up)
        # Short when price is significantly above VWAP (mean reversion down)
        long_entry = (price_deviation < -2.0 and vol_filter and vol_surge)
        short_entry = (price_deviation > 2.0 and vol_filter and vol_surge)
        
        # Exit conditions: price returns to within 0.5 std dev of VWAP
        exit_condition = abs(price_deviation) < 0.5
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position != 0 and exit_condition:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals