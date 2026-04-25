#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance zones. 
Breakouts above R3 or below S3, aligned with 12h EMA50 trend and confirmed by volume spikes,
capture strong momentum moves while avoiding counter-trend whipsaws. Discrete sizing (0.25) 
targets ~75-150 trades over 4 years to minimize fee drag. Uses ATR-based trailing stop for risk management.
Works in both bull (breakouts with trend) and bear (breakouts against trend but with volume confirmation).
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
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot levels (R3, S3, R4, S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR for stop loss (using 14 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for ATR (14) and to avoid NaN
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            i >= len(df_1d)):
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
        
        # Get previous completed 1d bar for Camarilla calculation
        if i < 1:
            continue
            
        # Use previous 1d bar's OHLC for pivot calculation (avoid look-ahead)
        prev_1d_idx = min(i // (24*4), len(df_1d)-1)  # approximate 1d bars in 4h
        if prev_1d_idx < 1:
            prev_1d_idx = 1
            
        # Ensure we don't use current forming 1d bar
        if prev_1d_idx >= len(df_1d):
            prev_1d_idx = len(df_1d) - 1
        if prev_1d_idx < 1:
            prev_1d_idx = 1
            
        # Calculate Camarilla levels from previous completed 1d bar
        try:
            prev_high = df_1d['high'].iloc[prev_1d_idx-1]
            prev_low = df_1d['low'].iloc[prev_1d_idx-1]
            prev_close = df_1d['close'].iloc[prev_1d_idx-1]
        except (IndexError, KeyError):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla pivot levels
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Key levels: R3, S3, R4, S4
        r3 = pivot + (range_val * 1.1 / 2)
        s3 = pivot - (range_val * 1.1 / 2)
        r4 = pivot + (range_val * 1.1)
        s4 = pivot - (range_val * 1.1)
        
        # Volume spike: current volume > 2.0 * 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > r3
        bearish_breakout = curr_close < s3
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.0*ATR from highest since entry
                if curr_close < highest_since_entry - 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout below S3
                elif curr_close < s3:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout above R3
                elif curr_close > r3:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume
        if position == 0:
            # Long: break above R3 AND price above 12h EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below S3 AND price below 12h EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
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

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0