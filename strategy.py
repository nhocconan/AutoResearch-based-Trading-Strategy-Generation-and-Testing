#!/usr/bin/env python3
"""
6h_cci_trend_volume_v1
Hypothesis: Use 12h CCI as trend filter and 6h CCI for entries with volume confirmation.
In trending markets (|CCI_12h| > 100), look for pullbacks in CCI_6h to enter with the trend.
Volume > 1.5x average confirms momentum. Works in both bull/bear by following trend.
Target: 20-40 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_trend_volume_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate CCI on 12h (20-period)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    ma_12h = typical_price_12h.rolling(window=20, min_periods=20).mean()
    mad_12h = typical_price_12h.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_12h = (typical_price_12h - ma_12h) / (0.015 * mad_12h)
    cci_12h = cci_12h.values
    
    # Calculate CCI on 6h (20-period) for entries
    typical_price = (high + low + close) / 3
    ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - ma) / (0.015 * mad)
    cci = cci.values
    
    # Align 12h CCI to 6h timeframe
    cci_12h_aligned = align_htf_to_ltf(prices, df_12h, cci_12h)
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(cci[i]) or np.isnan(cci_12h_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI_6h > 100 (overbought) or opposite signal
            if cci[i] > 100 or \
               (cci[i] < -100 and cci_12h_aligned[i] < -100 and volume[i] > 1.5 * avg_volume[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI_6h < -100 (oversold) or opposite signal
            if cci[i] < -100 or \
               (cci[i] > 100 and cci_12h_aligned[i] > 100 and volume[i] > 1.5 * avg_volume[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: |CCI_12h| > 100 indicates trend
            strong_uptrend = cci_12h_aligned[i] > 100
            strong_downtrend = cci_12h_aligned[i] < -100
            
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: pullback in uptrend (CCI_6h < -50) with volume
            if strong_uptrend and cci[i] < -50 and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: pullback in downtrend (CCI_6h > 50) with volume
            elif strong_downtrend and cci[i] > 50 and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals