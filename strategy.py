#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_RegimeFilter_v3
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and choppiness regime filter.
Long when: price breaks above Donchian upper + 1d EMA50 uptrend + volume > 2.0 * avg volume + chop > 61.8 (range).
Short when: price breaks below Donchian lower + 1d EMA50 downtrend + volume > 2.0 * avg volume + chop > 61.8.
Exit: ATR trailing stop (2.5 * ATR) or Donchian opposite touch.
Uses discrete 0.25 position size. Targets 20-35 trades/year for optimal generalization across BTC/ETH/SOL.
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
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # 1d HTF for trend and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index (CHOP) regime filter - using 14-period
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr * np.sqrt(atr_period) / (max_high - min_low)) / np.log10(atr_period)
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # default to neutral when range=0
    chop_regime = chop > 61.8  # ranging regime
    
    # ATR for trailing stop (2.5 * ATR)
    atr_stop_mult = 2.5
    atr_stop = atr * atr_stop_mult
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 20 for Donchian, 20 for volume avg, 50 for 1d EMA, 14 for ATR/CHOP
    start_idx = max(20, 20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_regime[i]) or np.isnan(atr_stop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and regime confirmation
            # Long: break above upper + 1d EMA50 uptrend + volume spike + chop > 61.8
            long_entry = (close_val > upper[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i] and \
                       chop_regime[i]
            # Short: break below lower + 1d EMA50 downtrend + volume spike + chop > 61.8
            short_entry = (close_val < lower[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_spike[i] and \
                        chop_regime[i]
            
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
            # Long - update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, close_val)
            
            # Exit conditions: ATR trailing stop OR Donchian lower touch
            long_exit = (close_val < highest_since_entry - atr_stop[i]) or \
                       (close_val < lower[i])
            
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions: ATR trailing stop OR Donchian upper touch
            short_exit = (close_val > lowest_since_entry + atr_stop[i]) or \
                        (close_val > upper[i])
            
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_RegimeFilter_v3"
timeframe = "4h"
leverage = 1.0