#!/usr/bin/env python3
"""
1d Donchian20 Breakout + 1w EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Daily Donchian(20) breakouts capture multi-day momentum. 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades. Volume spike confirms breakout strength. ATR-based stoploss manages risk. Works in bull/bear markets via trend filter and symmetric long/short logic. Target 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr = np.zeros(n)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_raw = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr = np.concatenate([[np.nan] * 14, atr_raw[14:]]) if len(atr_raw) > 14 else np.full(n, np.nan)
    
    # Start index: need enough for ATR and EMA warmup
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        atr_val = atr[i]
        
        # Donchian(20) channels: lookback 20 periods (excluding current)
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclude current bar
        if lookback_end - lookback_start < 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        highest_20 = np.max(high[lookback_start:lookback_end])
        lowest_20 = np.min(low[lookback_start:lookback_end])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND above 1w EMA50 (uptrend filter)
            long_condition = (curr_close > highest_20) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below Donchian(20) low AND below 1w EMA50 (downtrend filter)
            short_condition = (curr_close < lowest_20) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Stoploss: price drops to entry - 2.0 * ATR
            if curr_close <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit: price returns below Donchian(20) low or trend breaks
            elif curr_close < lowest_20 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Stoploss: price rises to entry + 2.0 * ATR
            if curr_close >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit: price returns above Donchian(20) high or trend breaks
            elif curr_close > highest_20 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0