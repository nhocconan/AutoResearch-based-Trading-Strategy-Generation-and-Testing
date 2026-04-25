#!/usr/bin/env python3
"""
12h Donchian20 Breakout + 1d EMA34 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian(20) breakouts on 12h capture multi-day momentum. 1d EMA34 filter ensures alignment with daily trend. Volume spike confirms institutional interest. ATR-based stoploss limits downside. Works in bull/bear markets via trend filter and symmetric long/short logic.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr = np.zeros(n)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Start index: need enough for EMA34 warmup and ATR
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        curr_atr = atr[i]
        
        # Donchian(20) channels
        if i >= 20:
            highest_20 = np.max(high[i-19:i+1])
            lowest_20 = np.min(low[i-19:i+1])
        else:
            highest_20 = np.max(high[:i+1])
            lowest_20 = np.min(low[:i+1])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 1d EMA34 (uptrend filter)
            long_condition = (curr_close > highest_20) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below Donchian lower AND below 1d EMA34 (downtrend filter)
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
            # Exit conditions: stoploss or trend break
            if curr_low <= entry_price - 2.5 * curr_atr or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: stoploss or trend break
            if curr_high >= entry_price + 2.5 * curr_atr or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0