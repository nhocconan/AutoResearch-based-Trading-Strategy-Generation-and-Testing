#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day volume-weighted VWAP as dynamic support/resistance.
# Price tends to revert to the prior day's VWAP, especially when touched with above-average volume.
# Long when price crosses above prior day VWAP with volume > 1.2x 20-period average, short when crosses below.
# Includes a 4-bar minimum holding period to reduce churn and a volatility filter (ATR > 0.5% of price) to avoid chop.
# Designed for low trade frequency (~20-30/year) with clear entry/exit rules.
# Works in bull/bear markets by capturing mean reversion to institutional VWAP levels.

name = "4h_1d_vwap_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate typical price and VWAP for each day
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align daily VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 20-period average volume for 4h timeframe
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    # Start from index 20 to ensure indicators are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr[i] > 0.005 * close[i]  # ATR > 0.5% of price
        
        # Volume filter: current volume > 1.2 * 20-period average
        vol_filter = vol_filter and (volume[i] > 1.2 * vol_avg_20[i])
        
        # Entry conditions: price crosses VWAP with volume
        long_entry = (close[i-1] <= vwap_aligned[i-1] and close[i] > vwap_aligned[i]) and vol_filter
        short_entry = (close[i-1] >= vwap_aligned[i-1] and close[i] < vwap_aligned[i]) and vol_filter
        
        # Exit conditions: minimum 4-bar hold or opposite VWAP cross
        if position != 0:
            bars_since_entry += 1
        
        time_exit = bars_since_entry >= 4
        opposite_cross = (position == 1 and close[i] < vwap_aligned[i]) or \
                         (position == -1 and close[i] > vwap_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            bars_since_entry = 0
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            bars_since_entry = 0
            signals[i] = -0.25
        elif position == 1 and (time_exit or opposite_cross):
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        elif position == -1 and (time_exit or opposite_cross):
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals