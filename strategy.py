#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: Trade 4h timeframe using Donchian channel (20) breakout for entry, 
daily EMA34 for trend filter, and daily volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price breaks above Donchian high AND above daily EMA34 AND volume spike. 
Enter short when price breaks below Donchian low AND below daily EMA34 AND volume spike. 
Exit on opposite Donchian touch or trend reversal. Uses discrete sizing 0.25 to balance 
return and drawdown. Target 20-50 trades/year on 4h timeframe. Works in bull/bear via 
price channel structure and trend filter. Avoids overtrading by requiring confluence 
of three conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily EMA34 trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Donchian channel (20-period) on primary 4h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA34 (34), volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above daily EMA34 AND volume spike
            long_setup = (close[i] > donchian_high[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below Donchian low AND below daily EMA34 AND volume spike
            short_setup = (close[i] < donchian_low[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian low OR closes below daily EMA34
            if (close[i] <= donchian_low[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high OR closes above daily EMA34
            if (close[i] >= donchian_high[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0