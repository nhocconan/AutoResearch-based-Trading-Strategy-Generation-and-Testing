#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: Daily Donchian(20) breakout confirmed by weekly EMA50 trend and volume spike, with ATR-based stoploss.
Works in bull markets via breakout continuation and in bear markets via short breakdowns.
Weekly trend filter prevents counter-trend trades. Volume spike ensures institutional participation.
ATR stoploss manages risk during false breakouts. Targets 15-25 trades/year for low fee drag.
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
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get weekly HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR(14) for stoploss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Warmup: need 20 for Donchian/volume, 50 for weekly EMA, 14 for ATR
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Position size: 25% of capital
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Donchian high + weekly EMA50 uptrend + volume spike
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below Donchian low + weekly EMA50 downtrend + volume spike
            short_entry = (close_val < donchian_low[i]) and \
                        (ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
                atr_at_entry = atr[i]
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
                atr_at_entry = atr[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on ATR-based stoploss or Donchian low break
            stop_price = entry_price - 2.5 * atr_at_entry
            if close_val < stop_price or close_val < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on ATR-based stoploss or Donchian high break
            stop_price = entry_price + 2.5 * atr_at_entry
            if close_val > stop_price or close_val > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0