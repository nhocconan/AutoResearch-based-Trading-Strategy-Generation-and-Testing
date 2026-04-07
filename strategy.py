#!/usr/bin/env python3
"""
6h_volatility_breakout_1d_trend_volume_v1
Hypothesis: On 6-hour timeframe, use 1-day ATR-based volatility breakout with volume confirmation and 1-day EMA trend filter. Breakouts from low volatility periods capture strong moves, while volume and trend filters avoid false breakouts. Works in both bull and bear regimes by capturing volatility expansion phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volatility_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6-day ATR average for volatility regime (low vol filter)
    atr_6d_avg = pd.Series(atr_14_1d_aligned).rolling(window=6, min_periods=6).mean().values
    
    # Volume filter: 24-period average on 6h timeframe (~6 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(30, 24), n):
        # Skip if data not available
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_6d_avg[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volatility filter: current ATR below average (low volatility setup)
        vol_low = atr_14_1d_aligned[i] < 0.8 * atr_6d_avg[i]
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA(50) or ATR expands too much (failed breakout)
            if close[i] < ema_50_1d_aligned[i] or atr_14_1d_aligned[i] > 2.0 * atr_6d_avg[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA(50) or ATR expands too much
            if close[i] > ema_50_1d_aligned[i] or atr_14_1d_aligned[i] > 2.0 * atr_6d_avg[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_low and vol_ok:
                # Breakout long: price breaks above EMA with upward momentum
                if close[i] > ema_50_1d_aligned[i] and close[i] > ema_50_1d_aligned[i-1]:
                    # Additional confirmation: strong close relative to range
                    if close[i] > (high[i] + low[i]) / 2:  # Close in upper half
                        position = 1
                        signals[i] = 0.25
                # Breakout short: price breaks below EMA with downward momentum
                elif close[i] < ema_50_1d_aligned[i] and close[i] < ema_50_1d_aligned[i-1]:
                    if close[i] < (high[i] + low[i]) / 2:  # Close in lower half
                        position = -1
                        signals[i] = -0.25
    
    return signals