#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Donchian(20) breakout on 4h with 1d EMA50 trend filter and volume spike confirmation.
Designed for 20-50 trades/year on BTC/ETH/SOL. Uses Donchian channels for structural breakouts,
1d EMA50 for stronger trend filter, and volume spike to confirm institutional participation.
Should work in bull markets (breakouts with volume) and bear markets (trend filter prevents wrong-way trades, volume fade false breakouts).
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h Donchian channels (20-period)
    if n < 20:
        return np.zeros(n)
    
    # Upper channel: highest high of last 20 periods
    high_series = pd.Series(high)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    
    # Lower channel: lowest low of last 20 periods
    low_series = pd.Series(low)
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period on 4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(50, 20, 14)  # EMA, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        ema_trend = ema_1d_aligned[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout in direction of 1d trend with volume confirmation
            # Long: price above 1d EMA50 AND break above upper Donchian + volume spike
            long_entry = (close_val > ema_trend) and (close_val > upper_channel[i]) and volume_spike[i]
            # Short: price below 1d EMA50 AND break below lower Donchian + volume spike
            short_entry = (close_val < ema_trend) and (close_val < lower_channel[i]) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on lower Donchian retracement or ATR stoploss
            exit_condition = (close_val < lower_channel[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on upper Donchian retracement or ATR stoploss
            exit_condition = (close_val > upper_channel[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0