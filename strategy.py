#!/usr/bin/env python3
# 12h_1d_vwap_std_dev_reversion_v1
# Strategy: 12h VWAP mean reversion with 1d standard deviation bands and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price deviating from VWAP by >2 standard deviations indicates overextension.
# In bull markets: buy dips to VWAP during uptrends. In bear markets: sell rallies to VWAP during downtrends.
# Volume confirmation ensures institutional participation. Low-frequency signals reduce fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vwap_std_dev_reversion_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h VWAP calculation (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / (vwap_den + 1e-10)
    
    # 12h price deviation from VWAP
    price_dev = close - vwap
    
    # 1d standard deviation of price deviations (20-period)
    price_dev_1d = np.zeros(len(df_1d))
    # Need to map 12h price_dev to 1d bars for std calculation
    # Create array of price deviations aligned to 12h but calculate std on 1d timeframe
    price_dev_series = pd.Series(price_dev)
    # For each 12h bar, we need the std of price_dev over the last 20 12h bars
    # But we want 1d std - so we'll use 2-period lookback (since 1d = 2x12h)
    price_dev_std = price_dev_series.rolling(window=2, min_periods=2).std().values  # 2*12h = 1d
    price_dev_std_20 = price_dev_series.rolling(window=20, min_periods=20).std().values  # 20*12h = 10d - too long
    # Correct approach: calculate std of price_dev over last 20 12h bars (~10 days) but we want shorter
    # Let's use 10-period (5 days) for more responsiveness
    price_dev_std = price_dev_series.rolling(window=10, min_periods=10).std().values
    
    # Upper and lower bands: VWAP ± 2 * std
    upper_band = vwap + 2.0 * price_dev_std
    lower_band = vwap - 2.0 * price_dev_std
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):  # Start after VWAP and std warmup
        # Skip if any required data is invalid
        if np.isnan(vwap[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Mean reversion signals
        price_below_lower = close[i] < lower_band[i]
        price_above_upper = close[i] > upper_band[i]
        price_below_vwap = close[i] < vwap[i]
        price_above_vwap = close[i] > vwap[i]
        
        # Entry conditions
        # Long: price below lower band AND volume confirmation
        if price_below_lower and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price above upper band AND volume confirmation
        elif price_above_upper and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to VWAP (mean reversion complete)
        elif position == 1 and price_above_vwap:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_below_vwap:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals