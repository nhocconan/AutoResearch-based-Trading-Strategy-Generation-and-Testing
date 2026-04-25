#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels represent stronger support/resistance than R1/S1.
Breakouts above H3 or below L3 with volume confirmation and 1d EMA34 trend alignment
capture strong momentum moves while filtering false breakouts. Works in bull/bear via trend filter.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d OHLC data for Camarilla calculation (using previous day's values)
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_open = df_1d['open'].values
    
    # Align 1d OHLC to 4h timeframe
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_close)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_high)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_low)
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_open)
    
    # Use previous day's OHLC for Camarilla calculation
    prev_close = close_1d_aligned
    prev_high = high_1d_aligned
    prev_low = low_1d_aligned
    
    # Calculate Camarilla levels using previous day's OHLC
    range_val = prev_high - prev_low
    camarilla_h3 = prev_close + range_val * 1.1 / 4
    camarilla_l3 = prev_close - range_val * 1.1 / 4
    camarilla_h4 = prev_close + range_val * 1.1 / 2
    camarilla_l4 = prev_close - range_val * 1.1 / 2
    
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
    
    # Start index: need enough for ATR (14) and EMA34 (34)
    start_idx = max(14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
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
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above H3 or below L3
        bullish_breakout = curr_close > h3
        bearish_breakout = curr_close < l3
        
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
                elif curr_close < l4 or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.5 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close > h4 or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume
        if position == 0:
            # Long: break above Camarilla H3 AND price above 1d EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below Camarilla L3 AND price below 1d EMA34
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

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0