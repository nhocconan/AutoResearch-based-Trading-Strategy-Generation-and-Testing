#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. 1w EMA50 filter ensures alignment with weekly trend (long in weekly uptrend, short in downtrend). Volume spike confirms institutional participation. Designed for 12h timeframe to limit trades (target: 12-37/year) and reduce fee drag. Works in both bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for volatility (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) - using prior 20 periods to avoid look-ahead
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + shift (1) + EMA50 (50)
    start_idx = 51  # 50 for EMA + 1 for Donchian shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        atr_value = atr[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above upper channel or below lower channel
        bullish_breakout = curr_close > upper_channel
        bearish_breakout = curr_close < lower_channel
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below lower channel or trend rejection (price below EMA)
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above upper channel or trend rejection (price above EMA)
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above upper channel AND price above 1w EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below lower channel AND price below 1w EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0