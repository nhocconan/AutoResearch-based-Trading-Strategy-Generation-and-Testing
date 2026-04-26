#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend_ATRStop_v1
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter, using ATR-based trailing stop. Targets 20-50 trades/year to minimize fee drag. Works in bull/bear markets via trend filter (only long in uptrend, short in downtrend) and volume confirmation to avoid false breakouts.
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # ATR for stoploss calculation
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 20 for Donchian, 50 for 1d EMA, 20 for volume avg, 14 for ATR
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.30  # 30% position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Donchian high + 1d EMA50 uptrend + volume confirmation
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_confirm[i]
            # Short: break below Donchian low + 1d EMA50 downtrend + volume confirmation
            short_entry = (close_val < donchian_low[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_confirm[i]
            
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
            # Long position - update highest and check ATR trailing stop
            highest_since_entry = max(highest_since_entry, close_val)
            # ATR trailing stop: exit if price drops 2.5 * ATR from highest point
            if close_val < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short position - update lowest and check ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.5 * ATR from lowest point
            if close_val > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeTrend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0