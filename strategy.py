#!/usr/bin/env python3
"""
1h_4d_VWAP_Reversion_v1
Hypothesis: Mean reversion from VWAP bands on 1h with 4h trend filter. 
In uptrend (4h price > VWAP), buy dips to VWAP; in downtrend (4h price < VWAP), sell rallies to VWAP.
Uses volume-weighted price to capture institutional interest. Designed for 20-40 trades/year with clear mean-reversion logic that works in both bull (buy dips) and bear (sell rallies) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_VWAP_Reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H VWAP FOR TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Typical price
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    # VWAP calculation
    vwap_4h = np.cumsum(typical_price_4h * volume_4h) / np.cumsum(volume_4h)
    # Handle division by zero at start
    vwap_4h = np.where(np.cumsum(volume_4h) != 0, vwap_4h, typical_price_4h)
    
    # Align VWAP to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # === 1H VWAP FOR ENTRY SIGNALS ===
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    vwap = np.where(np.cumsum(volume) != 0, vwap, typical_price)
    
    # Standard deviation of price-VWAP for bands
    price_dev = typical_price - vwap
    # Use rolling std with min_periods
    price_dev_series = pd.Series(price_dev)
    vwap_std = price_dev_series.rolling(window=20, min_periods=20).std().values
    
    # VWAP bands (1.5 standard deviations)
    vwap_upper = vwap + 1.5 * vwap_std
    vwap_lower = vwap - 1.5 * vwap_std
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(vwap[i]) or np.isnan(vwap_upper[i]) or np.isnan(vwap_lower[i]) or 
            np.isnan(vwap_4h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filter: 4h price relative to VWAP
        uptrend_4h = close_4h[i // 4] > vwap_4h[i // 4] if i // 4 < len(close_4h) else False
        downtrend_4h = close_4h[i // 4] < vwap_4h[i // 4] if i // 4 < len(close_4h) else False
        
        # Long: price touches VWAP lower band in uptrend
        long_signal = (close[i] <= vwap_lower[i] and 
                      uptrend_4h and 
                      volume[i] > 0)  # Ensure valid volume
        
        # Short: price touches VWAP upper band in downtrend
        short_signal = (close[i] >= vwap_upper[i] and 
                       downtrend_4h and 
                       volume[i] > 0)
        
        # Exit: price returns to VWAP
        exit_long = (position == 1 and close[i] >= vwap[i])
        exit_short = (position == -1 and close[i] <= vwap[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals