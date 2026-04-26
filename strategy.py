#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_VolumeSpike_HT
Hypothesis: On 6h timeframe, take Donchian(20) breakouts aligned with weekly trend (price > weekly EMA50 for long, < for short) and volume confirmation. Weekly trend provides strong directional filter that works in both bull (continuation) and bear (mean reversion off extreme deviations from weekly mean). Volume spike ensures breakout authenticity. Targets 12-25 trades/year (~50-100 over 4 years) to minimize fee drag while capturing significant moves.
"""

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
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on 6h
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 50 for weekly EMA, 20 for volume avg
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Conservative position size to manage drawdown
        
        if position == 0:
            # Flat - look for breakout with weekly trend and volume confirmation
            # Long: break above Donchian high + price > weekly EMA50 + volume spike
            long_entry = (close_val > donchian_high[i]) and \
                       (close_val > ema_50_1w_aligned[i]) and \
                       volume_spike[i]
            # Short: break below Donchian low + price < weekly EMA50 + volume spike
            short_entry = (close_val < donchian_low[i]) and \
                        (close_val < ema_50_1w_aligned[i]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price breaks below Donchian low (contrarian exit) or weekly trend turns bearish
            if (close_val < donchian_low[i]) or (close_val < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian high (contrarian exit) or weekly trend turns bullish
            if (close_val > donchian_high[i]) or (close_val > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0