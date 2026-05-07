#!/usr/bin/env python3
"""
4h_WilliamsVIX_Fix_Trend
Hypothesis: Williams VIX Fix identifies volatility spikes and mean reversion opportunities. 
Combined with 1d trend filter and volume confirmation, it captures reversals in both bull and bear markets.
The VIX Fix works by measuring how close the low is to the highest high over a period - 
in volatile markets, this spikes, signaling potential reversals. Using it as a contrarian 
signal with trend alignment reduces whipsaws. Target: 50-150 total trades over 4 years.
"""
name = "4h_WilliamsVIX_Fix_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams VIX Fix (22-period)
    # Measures put/call buying pressure - high values indicate fear
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    vix_fix = ((highest_high - low) / highest_high) * 100
    
    # VIX Fix signal: high values = fear = potential long opportunity
    # We'll use it inversely for mean reversion - when VIX Fix is high, consider long
    vix_fix_threshold = 60  # Empirical threshold for fear
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(vix_fix[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VIX Fix > threshold (fear) + price below 1d EMA (oversold in downtrend) + volume
            if vix_fix[i] > vix_fix_threshold and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: VIX Fix < threshold (low fear/complacency) + price above 1d EMA (overbought in uptrend) + volume
            elif vix_fix[i] < (vix_fix_threshold - 20) and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: VIX Fix returns to neutral levels or trend reversal
            if position == 1:
                if vix_fix[i] < (vix_fix_threshold - 10) or close[i] >= ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if vix_fix[i] > (vix_fix_threshold - 20) or close[i] <= ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals