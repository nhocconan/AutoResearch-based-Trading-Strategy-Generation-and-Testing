#!/usr/bin/env python3
"""
1h_4h_1d_Pullback_To_VWAP_With_Trend_Filter
Hypothesis: Uses 1d VWAP as dynamic support/resistance and 4h EMA20 trend filter.
Entries occur on 1h pullbacks to VWAP in direction of 4h trend, with volume confirmation.
Designed to work in both bull and bear markets by following 4h trend and mean-reverting to 1d VWAP.
Targets low trade frequency (15-30/year) via trend filter + VWAP mean reversion logic.
"""

name = "1h_4h_1d_Pullback_To_VWAP_With_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    vwap = np.cumsum(vwap_numerator) / np.cumsum(vwap_denominator)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d VWAP for Support/Resistance ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    vwap_1d = calculate_vwap(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        df_1d['volume'].values
    )
    vwap_1d_1h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # --- 4h EMA20 for Trend Filter ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # --- 1h VWAP for Entry Timing ---
    vwap_1h = calculate_vwap(high, low, close, volume)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_1h[i]) or np.isnan(ema_20_4h_1h[i]) or 
            np.isnan(vwap_1h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        # Trend direction from 4h EMA20
        trend = 1 if close[i] > ema_20_4h_1h[i] else -1
        
        if position == 0:
            # Long: uptrend + pullback to VWAP + volume
            if (trend == 1 and 
                close[i] > vwap_1h[i] and 
                close[i-1] <= vwap_1h[i-1] and  # crossed above VWAP this bar
                close[i] < vwap_1d_1h[i] and    # below 1d VWAP (pullback)
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: downtrend + bounce to VWAP + volume
            elif (trend == -1 and 
                  close[i] < vwap_1h[i] and 
                  close[i-1] >= vwap_1h[i-1] and  # crossed below VWAP this bar
                  close[i] > vwap_1d_1h[i] and    # above 1d VWAP (pullback)
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR price crosses above 1d VWAP
                if (trend == -1 or 
                    (close[i] > vwap_1d_1h[i] and close[i-1] <= vwap_1d_1h[i-1])):  # crossed above
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: trend turns up OR price crosses below 1d VWAP
                if (trend == 1 or 
                    (close[i] < vwap_1d_1h[i] and close[i-1] >= vwap_1d_1h[i-1])):  # crossed below
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals