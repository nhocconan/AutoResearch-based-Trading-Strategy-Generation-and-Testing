#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Donchian(20) breakouts capture strong momentum. 1d EMA34 filter ensures alignment with higher timeframe trend. Volume spike confirms breakout strength. Works in bull/bear via trend filter and discrete sizing (0.25) to limit fee drag (~50-150 trades over 4 years).
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
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volatility filtering (14 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20), EMA34 (34), ATR (14)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_value = atr[i]
        
        # Donchian channels (20-period)
        if i >= 20:
            highest_high = np.max(high[i-19:i+1])
            lowest_low = np.min(low[i-19:i+1])
        else:
            highest_high = np.max(high[:i+1])
            lowest_low = np.min(low[:i+1])
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > highest_high
        bearish_breakout = curr_close < lowest_low
        
        # Exit conditions: ATR-based trailing stop or opposite breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.5*ATR from highest high since entry
                if curr_close < highest_high - 2.5 * atr_value:
                    exit_signal = True
                # Opposite breakout or trend rejection
                elif bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.5*ATR from lowest low since entry
                if curr_close > lowest_low + 2.5 * atr_value:
                    exit_signal = True
                # Opposite breakout or trend rejection
                elif bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume
        if position == 0:
            # Long: break above upper Donchian AND price above 1d EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below lower Donchian AND price below 1d EMA34
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

name = "12h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0