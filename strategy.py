#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1
Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with 12h EMA50 trend filter and volume spike confirmation. Uses ATR-based stoploss and position sizing of 0.25. Designed to capture medium-term trends in both bull and bear markets by requiring HTF trend alignment and avoiding low-volume false breakouts. Target 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian(20) channels - use 4h data for breakout levels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR(14) for stoploss and position sizing reference
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), EMA50(12h), volume MA(20), ATR(14)
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_12h_val
        downtrend = close_val < ema_50_12h_val
        
        if position == 0:
            # Long: break above Donchian high with uptrend and volume spike
            long_signal = (high_val > highest_20[i]) and \
                          uptrend and \
                          vol_spike
            
            # Short: break below Donchian low with downtrend and volume spike
            short_signal = (low_val < lowest_20[i]) and \
                           downtrend and \
                           vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # ATR-based stoploss: exit if price drops 2.5 * ATR below entry
            if close_val < entry_price - (2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # ATR-based stoploss: exit if price rises 2.5 * ATR above entry
            if close_val > entry_price + (2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0