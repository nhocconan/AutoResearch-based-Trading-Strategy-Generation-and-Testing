#!/usr/bin/env python3
# 6h_1d_adaptive_cci_volume_momentum
# Hypothesis: 6-hour CCI (20) with 1d volume momentum filter
# Uses CCI for overbought/oversold detection with volume confirmation to filter false signals
# Works in bull/bear by adapting to volatility and using volume as confirmation of conviction
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_1d_adaptive_cci_volume_momentum"
timeframe = "6h"
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
    
    # Get daily data for volume momentum calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # CCI calculation (20-period)
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    # Avoid division by zero
    cci = np.where(mad != 0, (tp - ma_tp) / (0.015 * mad), 0.0)
    
    # Volume momentum: 1d volume ratio (current vs 20-day average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_momentum = np.where(vol_ma_1d != 0, volume_1d / vol_ma_1d, 1.0)
    vol_momentum_aligned = align_htf_to_ltf(prices, df_1d, vol_momentum)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(cci[i]) or np.isnan(vol_momentum_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: CCI crosses above -100 (oversold recovery) with volume confirmation
        if (cci[i] > -100 and cci[i-1] <= -100 and 
            vol_momentum_aligned[i] > 1.2 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: CCI crosses below 100 (overbought rejection) with volume confirmation
        elif (cci[i] < 100 and cci[i-1] >= 100 and 
              vol_momentum_aligned[i] > 1.2 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: CCI crosses zero line (mean reversion)
        elif position == 1 and cci[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals