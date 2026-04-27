#!/usr/bin/env python3
"""
6h_WilliamsVixFix_Reversion_1wTrend_VolumeSpike
Hypothesis: Williams Vix Fix identifies volatility spikes (fear/greed extremes) on 6h. 
Enter mean-reversion trades when Vix Fix > 0.8 (extreme fear) in 1w uptrend (long) or 
Vix Fix > 0.8 in 1w downtrend (short), with volume confirmation. 
Exit when Vix Fix < 0.5 (normalized volatility). 
Works in bull/bear markets by using 1w trend for direction and Vix Fix for timing mean reversion spikes.
Target: 50-150 trades over 4 years (12-37/year) with low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams Vix Fix calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Vix Fix on 6h: measures volatility spikes
    # Vix Fix = (Highest Close in lookback - Low) / (Highest Close in lookback - Lowest Low in lookback) * 100
    lookback = 22  # ~1 month of 6h bars (22*6h = 132h ~ 5.5 days)
    highest_close = pd.Series(df_6h['close'].values).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_6h['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    vix_fix = (highest_close - df_6h['low'].values) / (highest_close - lowest_low) * 100
    # Normalize to 0-1 range for easier thresholds
    vix_fix = vix_fix / 100.0
    
    # Align Vix Fix to 6h timeframe (already on 6h, but using align for consistency)
    vix_fix_aligned = align_htf_to_ltf(prices, df_6h, vix_fix)
    
    # 1w EMA34 for trend filter (slower, more reliable trend)
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Vix Fix lookback (22), 1w EMA34 (34), volume avg (20)
    start_idx = max(34, 22, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vix_fix_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vix_val = vix_fix_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: extreme Vix Fix (volatility spike) with 1w trend filter AND volume
            # Long: Vix Fix > 0.8 (extreme fear) AND 1w uptrend AND volume confirmation
            long_condition = (vix_val > 0.8) and (close_val > ema_val) and vol_conf
            # Short: Vix Fix > 0.8 (extreme fear/greed) AND 1w downtrend AND volume confirmation
            short_condition = (vix_val > 0.8) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when Vix Fix normalizes (< 0.5) OR trend breaks
            exit_condition = (vix_val < 0.5) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when Vix Fix normalizes (< 0.5) OR trend breaks
            exit_condition = (vix_val < 0.5) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsVixFix_Reversion_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0