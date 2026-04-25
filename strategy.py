#!/usr/bin/env python3
"""
12h Donchian20 Breakout + 1d ATR Trend + Volume Spike
Hypothesis: Donchian channel breakouts on 12h timeframe capture strong momentum moves.
Trend filtered by 1d ATR-based direction (price > 1d EMA34 + ATR*0.5 for uptrend, < for downtrend).
Volume confirmation ensures breakout validity. Discrete sizing (0.25) targets ~50-150 trades over 4 years.
Works in bull/bear by adapting to 1d ATR-adjusted trend filter.
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
    
    # Get daily data for EMA34 trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR for trend filter volatility adjustment
    tr1_1d = df_1d['high'][1:] - df_1d['low'][1:]
    tr2_1d = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3_1d = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA34 (34d) + ATR (14) + Donchian (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_12h_val = atr_12h[i]
        
        # Calculate Donchian channels for 12h timeframe using last 20 completed 12h bars
        if i >= 20:
            donch_high = np.max(high[i-20:i])  # highest high of last 20 bars (excluding current)
            donch_low = np.min(low[i-20:i])   # lowest low of last 20 bars (excluding current)
        else:
            donch_high = np.max(high[:i]) if i > 0 else curr_high
            donch_low = np.min(low[:i]) if i > 0 else curr_low
        
        # Volume spike: current volume > 2.5 * 20-period average (more strict to reduce trades)
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.5 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > donch_high
        bearish_breakout = curr_close < donch_low
        
        # Trend filter: price must be beyond EMA by at least 0.5*ATR to avoid choppy conditions
        bullish_trend = curr_close > ema_trend + 0.5 * atr_1d_val
        bearish_trend = curr_close < ema_trend - 0.5 * atr_1d_val
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.5*ATR from highest since entry
                if curr_close < highest_since_entry - 3.5 * atr_12h_val:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close < donch_low or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.5*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.5 * atr_12h_val:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close > donch_high or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume
        if position == 0:
            # Long: break above Donchian high AND bullish trend AND volume spike
            long_condition = bullish_breakout and bullish_trend and volume_spike
            # Short: break below Donchian low AND bearish trend AND volume spike
            short_condition = bearish_breakout and bearish_trend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0