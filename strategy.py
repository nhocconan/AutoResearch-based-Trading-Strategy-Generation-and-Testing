#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when: price breaks above 20-period high + 12h EMA50 uptrend + volume > 2.0 * avg volume.
Short when: price breaks below 20-period low + 12h EMA50 downtrend + volume > 2.0 * avg volume.
Exit via ATR-based trailing stop (3*ATR) or opposite Donchian break.
Uses discrete 0.25 position size. Targets 30-50 trades/year for optimal test generalization.
Works in both bull (trend continuation) and bear (mean reversion via tight stops).
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for trailing stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 20 for Donchian/volume, 50 for 12h EMA, 14 for ATR
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Donchian high + 12h EMA50 uptrend + volume spike
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below Donchian low + 12h EMA50 downtrend + volume spike
            short_entry = (close_val < donchian_low[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - update highest and check trailing stop
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit if: price drops 3*ATR from high OR breaks below Donchian low
            trailing_stop = highest_since_entry - (3.0 * atr[i])
            if (close_val < trailing_stop) or (close_val < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - update lowest and check trailing stop
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit if: price rises 3*ATR from low OR breaks above Donchian high
            trailing_stop = lowest_since_entry + (3.0 * atr[i])
            if (close_val > trailing_stop) or (close_val > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0