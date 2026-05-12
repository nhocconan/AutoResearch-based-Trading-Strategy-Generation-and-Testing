#!/usr/bin/env python3
name = "4h_StructureBreak_MR_Zscore_Volume"
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
    
    # Daily close for structure and mean-reversion
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 20-period Z-score of daily returns (mean reversion signal)
    returns_1d = np.diff(np.log(close_1d), prepend=np.log(close_1d[0]))
    mean_20 = pd.Series(returns_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(returns_1d).rolling(window=20, min_periods=20).std().values
    zscore_20 = (returns_1d - mean_20) / (std_20 + 1e-9)
    zscore_20_aligned = align_htf_to_ltf(prices, df_1d, zscore_20)
    
    # Daily volume filter: volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Daily structure: 20-period high/low for breakout
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Session filter: active during London/NY overlap (08-16 UTC) and Asia (00-08 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if daily data not ready
        if np.isnan(zscore_20_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or \
           np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: active during London/NY overlap (08-16 UTC) and Asia (00-08 UTC)
        hour = hours[i]
        in_session = ((0 <= hour <= 8) or (8 <= hour <= 16))
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above daily 20-period high AND Z-score < -1.5 (oversold) with volume confirmation
            if (high[i] > high_20_1d_aligned[i] and 
                close[i] > high_20_1d_aligned[i] and
                zscore_20_aligned[i] < -1.5 and
                volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily 20-period low AND Z-score > 1.5 (overbought) with volume confirmation
            elif (low[i] < low_20_1d_aligned[i] and 
                  close[i] < low_20_1d_aligned[i] and
                  zscore_20_aligned[i] > 1.5 and
                  volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below daily 20-period low or Z-score reverts
            if (low[i] < low_20_1d_aligned[i] or 
                zscore_20_aligned[i] > -0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above daily 20-period high or Z-score reverts
            if (high[i] > high_20_1d_aligned[i] or 
                zscore_20_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals