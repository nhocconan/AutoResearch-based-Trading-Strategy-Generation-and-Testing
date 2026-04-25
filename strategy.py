#!/usr/bin/env python3
"""
1h Volume Spike + 4h EMA Trend + Daily Session Filter
Hypothesis: Volume spikes on 1h indicate institutional interest. Combined with 4h EMA trend filter and active session (08-20 UTC), captures momentum moves while avoiding low-liquidity periods. Designed for 15-30 trades/year/symbol to minimize fee drag. Works in bull (trend continuation) and bear (mean reversion after extreme spikes) markets.
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
    open_time = prices['open_time']
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # ATR for stop sizing
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 4h EMA50 trend filter (MTF) - loaded ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily EMA200 for long-term bias filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 200) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: volume spike + price above/below EMAs
            # Long: price > 4h EMA50 AND price > daily EMA200 (bullish alignment)
            # Short: price < 4h EMA50 AND price < daily EMA200 (bearish alignment)
            long_entry = vol_spike and (curr_close > ema_50_4h_aligned[i]) and (curr_close > ema_200_1d_aligned[i])
            short_entry = vol_spike and (curr_close < ema_50_4h_aligned[i]) and (curr_close < ema_200_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: trail with 2.5 * ATR
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: trail with 2.5 * ATR
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_4hEMA50_1dEMA200_SessionFilter"
timeframe = "1h"
leverage = 1.0