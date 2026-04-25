#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Donchian breakouts capture momentum. 12h EMA50 filters trend direction. Volume spike confirms participation. Works in bull/bear by trend-following. Target: 20-50 trades/year (80-200 over 4 years).
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian (20) + EMA50 (50) + ATR (14)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        atr_value = atr[i]
        
        # Donchian channels (20-period)
        if i >= 20:
            donch_high = np.max(high[i-19:i+1])
            donch_low = np.min(low[i-19:i+1])
        else:
            donch_high = np.max(high[:i+1])
            donch_low = np.min(low[:i+1])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > donch_high
        bearish_breakout = curr_close < donch_low
        
        # Stoploss: ATR-based
        if position != 0:
            stop_signal = False
            if position == 1:
                # Long stop: price < entry_price - 2.0 * ATR
                if curr_close < entry_price - 2.0 * atr_value:
                    stop_signal = True
            elif position == -1:
                # Short stop: price > entry_price + 2.0 * ATR
                if curr_close > entry_price + 2.0 * atr_value:
                    stop_signal = True
            
            if stop_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above Donchian high AND price above 12h EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below Donchian low AND price below 12h EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0