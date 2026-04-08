# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_12h_vwap_crossover_v1
Hypothesis: VWAP crossovers on 4h timeframe with 12h trend filter and volume confirmation.
Works in both bull and bear markets by capturing mean-reversion to VWAP during trends.
- Long: 4h VWAP crosses above price AND 12h trend up AND volume > 1.5x average
- Short: 4h VWAP crosses below price AND 12h trend down AND volume > 1.5x average
- Exit: VWAP crossover in opposite direction
Uses VWAP as dynamic support/resistance, reducing whipsaws in choppy markets.
Target: 25-40 trades/year to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_vwap_crossover_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tp_volume = typical_price * volume
    
    # Cumulative VWAP (reset each day)
    cum_tp_volume = np.zeros(n)
    cum_volume = np.zeros(n)
    vwap = np.full(n, np.nan)
    
    # Reset cumulative values at daily boundaries
    for i in range(n):
        if i == 0 or prices['open_time'].iloc[i].date() != prices['open_time'].iloc[i-1].date():
            cum_tp_volume[i] = tp_volume[i]
            cum_volume[i] = volume[i]
        else:
            cum_tp_volume[i] = cum_tp_volume[i-1] + tp_volume[i]
            cum_volume[i] = cum_volume[i-1] + volume[i]
        
        if cum_volume[i] > 0:
            vwap[i] = cum_tp_volume[i] / cum_volume[i]
    
    # Calculate average volume for confirmation
    vol_avg = np.zeros(n)
    vol_avg[19] = np.mean(volume[:20])  # 20-period average
    for i in range(20, n):
        vol_avg[i] = (vol_avg[i-1] * 19 + volume[i]) / 20
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Simple trend: price above/below 20-period SMA
    sma_12h_20 = np.zeros(len(close_12h))
    sma_12h_20[:] = np.nan
    sma_12h_20[19] = np.mean(close_12h[:20])
    for i in range(20, len(close_12h)):
        sma_12h_20[i] = (sma_12h_20[i-1] * 19 + close_12h[i]) / 20
    
    sma_12h_20_aligned = align_htf_to_ltf(prices, df_12h, sma_12h_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(vwap[i]) or np.isnan(sma_12h_20_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        trend_12h = sma_12h_20_aligned[i]
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
        
        if position == 1:  # Long
            # Exit: price crosses below VWAP OR trend turns down
            if price < vwap_val or close[i] < trend_12h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above VWAP OR trend turns up
            if price > vwap_val or close[i] > trend_12h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions with volume confirmation
            if vol_ratio > 1.5:  # Volume confirmation
                # Long: price crosses above VWAP AND trend up
                if price > vwap_val and close[i] > trend_12h:
                    # Confirm crossover: previous bar was below VWAP
                    if i > 0 and close[i-1] <= vwap[i-1]:
                        position = 1
                        signals[i] = 0.25
                # Short: price crosses below VWAP AND trend down
                elif price < vwap_val and close[i] < trend_12h:
                    # Confirm crossover: previous bar was above VWAP
                    if i > 0 and close[i-1] >= vwap[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals