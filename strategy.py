#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h 12-period ATR breakout with daily volume confirmation
# Hypothesis: ATR breakouts capture volatility expansion moves; volume confirms institutional participation.
# Works in bull via upward breakouts, in bear via downward breakdowns. ATR adapts to volatility regime.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_atr12_1d_volume_breakout_v1"
timeframe = "4h"
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
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(12) for breakout threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(12, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below open - ATR (mean reversion signal)
            if close[i] < (prices['open'].iloc[i] - atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above open + ATR (mean reversion signal)
            if close[i] > (prices['open'].iloc[i] + atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above open + ATR + volume confirmation
            if close[i] > (prices['open'].iloc[i] + atr[i]) and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below open - ATR + volume confirmation
            elif close[i] < (prices['open'].iloc[i] - atr[i]) and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals