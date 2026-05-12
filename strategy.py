#!/usr/bin/env python3
name = "12h_RVOL_MeanReversion_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1d data for trend filter and RVOL calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # RVOL (Relative Volume): current volume / average volume over last 20 periods
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / vol_avg_20
    
    # Z-score of RVOL over 50 periods to identify extreme volume spikes
    rvol_mean = pd.Series(rvol).rolling(window=50, min_periods=50).mean().values
    rvol_std = pd.Series(rvol).rolling(window=50, min_periods=50).std().values
    rvol_z = (rvol - rvol_mean) / rvol_std
    # Replace division by zero or NaN with 0
    rvol_z = np.where((rvol_std == 0) | np.isnan(rvol_std), 0, rvol_z)
    
    # Mean reversion signal: extreme RVOL spike suggests mean reversion opportunity
    # Only trade when RVOL z-score exceeds 2.0 (significant spike)
    vol_spike = rvol_z > 2.0
    
    # Price deviation from 1d EMA34: look for mean reversion to trend
    price_dev = (close - ema_34_1d_aligned) / ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rvol_z[i]) or np.isnan(price_dev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below trend AND significant volume spike (oversold bounce)
            if price_dev[i] < -0.015 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above trend AND significant volume spike (overbought rejection)
            elif price_dev[i] > 0.015 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to trend or opposite volume spike
            if price_dev[i] > -0.005 or (price_dev[i] > 0 and vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to trend or opposite volume spike
            if price_dev[i] < 0.005 or (price_dev[i] < 0 and vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals