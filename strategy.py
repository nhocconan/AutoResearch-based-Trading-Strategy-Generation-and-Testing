#!/usr/bin/env python3
# Hypothesis: 4h price action relative to 12h VWAP with volume confirmation and ATR-based risk management.
# Uses 12h VWAP as dynamic support/resistance - price above VWAP indicates bullish bias, below indicates bearish bias.
# Entry when price closes decisively above/below 12h VWAP with above-average volume.
# Exit when price crosses back through VWAP. Designed for low trade frequency (<25/year) to minimize fee drag.
# VWAP acts as institutional reference point, providing edge in both trending and ranging markets.

name = "4h_VWAP12h_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h VWAP data once before loop
    df_12h = get_htf_data(prices, '12h')
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_raw = (typical_price * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h = align_htf_to_ltf(prices, df_12h, vwap_raw.values)
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(vwap_12h[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above 12h VWAP with volume confirmation
            if close[i] > vwap_12h[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below 12h VWAP with volume confirmation
            elif close[i] < vwap_12h[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back below 12h VWAP
            if close[i] < vwap_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back above 12h VWAP
            if close[i] > vwap_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals