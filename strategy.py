#!/usr/bin/env python3
"""
6h_IBS_MeanReversion_1dTrendFilter_VolumeSpike_v1
Hypothesis: On 6h timeframe, use Intraday Bar Statistic (IBS = (Close-Low)/(High-Low)) for mean reversion entries during extreme oversold/overbought conditions, filtered by 1d trend direction (price > EMA50) and volume spike confirmation. In bull trends (price > 1d EMA50), take long IBS < 0.15; in bear trends (price < 1d EMA50), take short IBS > 0.85. Volume spike ensures institutional participation. Designed for 12-30 trades/year to avoid fee drag while working in both bull and bear regimes via trend-filtered mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # IBS calculation for 6h: (Close - Low) / (High - Low)
    ibs = (close - low) / (high - low)
    ibs = np.where((high - low) == 0, 0.5, ibs)  # avoid division by zero
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for EMA50 and volume average
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ibs[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ibs_val = ibs[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for mean reversion entry with trend filter and volume spike
            # Long: IBS < 0.15 (oversold) AND price > 1d EMA50 (bull trend) AND volume spike
            long_entry = (ibs_val < 0.15) and (close_val > ema_50_val) and vol_spike
            # Short: IBS > 0.85 (overbought) AND price < 1d EMA50 (bear trend) AND volume spike
            short_entry = (ibs_val > 0.85) and (close_val < ema_50_val) and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when IBS > 0.5 (mean reversion complete) or reverse signal
            if ibs_val > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when IBS < 0.5 (mean reversion complete) or reverse signal
            if ibs_val < 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_IBS_MeanReversion_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0