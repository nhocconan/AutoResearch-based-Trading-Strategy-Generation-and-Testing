#!/usr/bin/env python3
# 6h_1d_cci_rvol_mean_reversion_v1
# Strategy: 6-hour Commodity Channel Index (CCI) mean reversion with 1-day volume filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions on 6h timeframe. 
# Mean reversion trades are taken when CCI crosses back from extreme levels (>100 or <-100) 
# with confirmation from elevated 1-day relative volume (RVOL > 1.5) to filter for institutional interest.
# Works in both bull and bear markets as mean reversion occurs during pullbacks in trends and 
# during range-bound periods. Volume filter ensures trades occur with participation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_rvol_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume average for RVOL calculation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 6h CCI (20-period) for mean reversion signals
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    ma_tp = tp_series.rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation
    mad = tp_series.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # Avoid division by zero
    cci = (typical_price - ma_tp) / (0.015 * mad + 1e-10)
    
    # Daily relative volume: current day volume / 20-day average volume
    # Note: we use the current day's volume aligned to 6s bars
    vol_1d_current = df_1d['volume'].values
    vol_1d_current_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
    rvol_1d = vol_1d_current_aligned / (vol_avg_20_1d_aligned + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after CCI warmup
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(rvol_1d[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion conditions
        # Long: CCI crosses above -100 from below (oversold bounce)
        long_signal = (cci[i-1] <= -100) and (cci[i] > -100)
        # Short: CCI crosses below 100 from above (overbought reversal)
        short_signal = (cci[i-1] >= 100) and (cci[i] < 100)
        
        # Volume filter: elevated daily volume suggests participation
        vol_filter = rvol_1d[i] > 1.5
        
        # Entry logic: mean reversion + volume confirmation
        if long_signal and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite mean reversion signal
        elif position == 1 and short_signal:
            position = 0
            signals[i] = 0.0
        elif position == -1 and long_signal:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals