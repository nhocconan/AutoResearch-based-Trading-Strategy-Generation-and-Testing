#!/usr/bin/env python3
"""
12h_vwap_1d_trend_volume_v1
Hypothesis: VWAP acts as dynamic support/resistance on 12h chart.
Price reversals at VWAP with volume confirmation and trend alignment capture mean reversion moves.
Works in both bull and bear markets by trading in direction of 1d EMA50 trend.
Targets 15-40 trades/year with disciplined entries to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_vwap_1d_trend_volume_v1"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, np.nan)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP OR trend turns down
            if close[i] < vwap[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP OR trend turns up
            if close[i] > vwap[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches VWAP from below + volume confirmation + uptrend
            if (close[i] >= vwap[i] and 
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches VWAP from above + volume confirmation + downtrend
            elif (close[i] <= vwap[i] and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals