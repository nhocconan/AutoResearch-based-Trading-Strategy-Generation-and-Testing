#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_v1
Hypothesis: Trade weekly Donchian channel breakouts on daily timeframe with volume confirmation.
Only long when price breaks above weekly Donchian high (20-period) in bullish regime (price > weekly EMA50),
short when breaks below weekly Donchian low in bearish regime (price < weekly EMA50).
Volume spike > 2.0 * ATR20 confirms momentum. Uses discrete sizing 0.25 to minimize fee drag.
Designed for low trade frequency (target: 15-25 trades/year) to avoid overtrading while capturing strong directional moves.
Works in bull via breakouts, works in bear via short breakdowns, avoids ranging markets via regime filter.
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
    
    # Get weekly data for Donchian channels and trend regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need enough for EMA50 and Donchian20
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend regime
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: max(high, lookback=20)
    dh_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: min(low, lookback=20)
    dl_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_1w, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1w, dl_20)
    
    # Calculate ATR for volume spike filter (using weekly ATR20)
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - df_1w['close'].values[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - df_1w['close'].values[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr_1w = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 (50) and Donchian (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(dh_20_aligned[i]) or 
            np.isnan(dl_20_aligned[i]) or np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.0 * weekly ATR
        volume_spike = volume[i] > 2.0 * atr_1w_aligned[i]
        
        # Determine weekly trend regime
        # Bull regime: price > weekly EMA50
        # Bear regime: price < weekly EMA50
        if close[i] > ema_50_1w_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_50_1w_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades (unlikely but handle)
        
        if position == 0:
            # Long setup: price breaks above weekly Donchian high AND volume spike AND bull regime
            long_setup = (close[i] > dh_20_aligned[i]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below weekly Donchian low AND volume spike AND bear regime
            short_setup = (close[i] < dl_20_aligned[i]) and volume_spike and (regime == 'bear')
            
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
            # Exit: price breaks below weekly Donchian low OR regime turns bearish
            if (close[i] < dl_20_aligned[i]) or (regime == 'bear'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above weekly Donchian high OR regime turns bullish
            if (close[i] > dh_20_aligned[i]) or (regime == 'bull'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchianBreakout_v1"
timeframe = "1d"
leverage = 1.0