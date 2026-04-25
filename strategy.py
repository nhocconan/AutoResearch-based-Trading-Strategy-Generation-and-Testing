#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with 12h EMA50 Trend and Volume Spike
Hypothesis: Camarilla R1/S1 levels provide high-probability intraday support/resistance on 4h charts.
Breakouts above R1 or below S1 with volume confirmation and 12h EMA50 trend alignment capture
strong momentum moves while filtering false breakouts. Works in bull/bear via trend filter.
Discrete sizing (0.30) targets ~75-150 trades over 4 years to minimize fee drag.
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
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1d OHLC data
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_open = df_1d['open'].values
    
    # Align 1d OHLC to 4h timeframe (using previous day's values)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_close)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_high)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_low)
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_open)
    
    # Use previous day's OHLC for Camarilla calculation (shift by 1 bar to avoid look-ahead)
    prev_close = close_1d_aligned  # This will be shifted inside align_htf_to_ltf to previous day's close
    prev_high = high_1d_aligned
    prev_low = low_1d_aligned
    
    # Calculate Camarilla levels using previous day's OHLC
    range_val = prev_high - prev_low
    camarilla_r1 = prev_close + range_val * 1.1 / 12
    camarilla_s1 = prev_close - range_val * 1.1 / 12
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    
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
    
    # Start index: need enough for ATR (14) and EMA50 (50)
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
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
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above R1 or below S1
        bullish_breakout = curr_close > r1
        bearish_breakout = curr_close < s1
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.5*ATR from highest since entry
                if curr_close < highest_since_entry - 2.5 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close < s3 or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.5 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close > r3 or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume
        if position == 0:
            # Long: break above Camarilla R1 AND price above 12h EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below Camarilla S1 AND price below 12h EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.30
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0