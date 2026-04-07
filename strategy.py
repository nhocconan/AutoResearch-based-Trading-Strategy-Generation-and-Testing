#!/usr/bin/env python3
"""
4h_volatility_breakout_12h_ema_v1
Hypothesis: Price breaks above/below 12h EMA during low volatility periods (Bollinger Band squeeze) with volume confirmation capture institutional breakouts.
Works in both bull/bear markets by using volatility regime filter - only trade when volatility is low (squeeze) and breaks with volume.
Targets 20-40 trades/year to minimize fee drag while capturing strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volatility_breakout_12h_ema_v1"
timeframe = "4h"
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
    
    # 12h EMA for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Bollinger Bands for volatility regime (20-period, 2 std)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile for squeeze detection (low volatility)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=30).apply(
        lambda x: np.percentile(x, 25) if len(x) > 0 else 0, raw=True
    ).values
    volatility_squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bb_width[i]) or np.isnan(volatility_squeeze[i]) or
            vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        vol_confirmed = volume[i] > vol_ma[i] * 1.5  # Require 1.5x average volume
        
        if position == 1:  # Long position
            # Exit: price closes below 12h EMA or volatility expands significantly
            if close[i] < ema_12h_aligned[i] or bb_width[i] > bb_width_percentile[i] * 2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h EMA or volatility expands significantly
            if close[i] > ema_12h_aligned[i] or bb_width[i] > bb_width_percentile[i] * 2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volatility breakout long: price breaks above EMA during low volatility with volume
            if (close[i] > ema_12h_aligned[i] and 
                volatility_squeeze[i] and 
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Volatility breakout short: price breaks below EMA during low volatility with volume
            elif (close[i] < ema_12h_aligned[i] and 
                  volatility_squeeze[i] and 
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals