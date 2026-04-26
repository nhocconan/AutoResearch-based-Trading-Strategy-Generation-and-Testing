#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_Regime
Hypothesis: Donchian(20) breakout with volume spike confirmation and 1d EMA34 trend filter.
Long when: price breaks above Donchian(20) high + volume > 1.5*avg_volume(20) + close > 1d EMA34.
Short when: price breaks below Donchian(20) low + volume > 1.5*avg_volume(20) + close < 1d EMA34.
Exit: opposite Donchian breakout or volume drops below average.
Uses 4h timeframe with discrete 0.25 position size. Designed for BTC/ETH:
- Captures strong trending moves with volume confirmation
- 1d EMA34 filter avoids counter-trend trades in bear markets
- Volume spike reduces false breakouts
- Targets 20-40 trades/year for optimal risk/reward.
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
    
    # Donchian(20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 1d EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian/volume, 34 for EMA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for breakout with volume spike and HTF trend alignment
            long_breakout = close_val > donch_high[i]
            short_breakout = close_val < donch_low[i]
            
            if long_breakout and vol_spike and (close_val > ema_34_aligned[i]):
                signals[i] = size
                position = 1
            elif short_breakout and vol_spike and (close_val < ema_34_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on opposite breakout or volume drops below average
            if close_val < donch_low[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on opposite breakout or volume drops below average
            if close_val > donch_high[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_Regime"
timeframe = "4h"
leverage = 1.0