#!/usr/bin/env python3
"""
Hypothesis: 12h VWAP mean reversion with weekly trend filter.
In bull markets (price > weekly VWAP): long when price touches VWAP with volume confirmation.
In bear markets (price < weekly VWAP): short when price touches VWAP with volume confirmation.
Weekly VWAP acts as dynamic support/resistance. Volume confirms institutional interest.
Designed for low trade frequency (12-37/year) with clear entry/exit rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_vwap_mean_reversion_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # === WEEKLY VWAP (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate VWAP: cumulative(volume * typical_price) / cumulative(volume)
    weekly_typical = (weekly_high + weekly_low + weekly_close) / 3
    weekly_vwap_values = np.cumsum(weekly_volume * weekly_typical) / np.cumsum(weekly_volume)
    weekly_vwap_values = np.where(np.cumsum(weekly_volume) == 0, 0, weekly_vwap_values)
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap_values)
    
    # === WEEKLY TREND (price vs VWAP) ===
    weekly_trend_up = close > weekly_vwap_aligned
    
    # === 12H VWAP (LTF) ===
    typical_price = (high + low + close) / 3
    vwap_values = np.cumsum(volume * typical_price) / np.cumsum(volume)
    vwap_values = np.where(np.cumsum(volume) == 0, 0, vwap_values)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        if np.isnan(weekly_vwap_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # Long position
            if close[i] < vwap[i] or not weekly_trend_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if close[i] > vwap[i] or weekly_trend_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: price touches VWAP (within 0.1%) with volume
            price_to_vwap_ratio = close[i] / vwap[i]
            if 0.999 <= price_to_vwap_ratio <= 1.001:  # Within 0.1% of VWAP
                if weekly_trend_up[i]:  # Bull trend: long at VWAP support
                    position = 1
                    signals[i] = 0.25
                else:  # Bear trend: short at VWAP resistance
                    position = -1
                    signals[i] = -0.25
    
    return signals