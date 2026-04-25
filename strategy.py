#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong intraday resistance/support on 12h timeframe.
Breakouts above H3 or below L3 with volume confirmation and aligned with 1d EMA34 trend
capture momentum moves while avoiding counter-trend whipsaws. Discrete sizing (0.25)
targets ~50-150 trades over 4 years to minimize fee drag. Works in bull/bear by
adapting to 1d EMA34 direction.
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
    open_price = prices['open'].values
    
    # Get daily data for EMA34 trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA34 (34d) + ATR (14)
    start_idx = max(34, 14)
    
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
        curr_open = open_price[i]
        ema_trend = ema_34_aligned[i]
        atr_value = atr[i]
        
        # Calculate Camarilla levels for 12h timeframe using previous 12h bar
        if i >= 1:
            # Use previous completed 12h bar for Camarilla calculation
            phigh = high[i-1]
            plow = low[i-1]
            pclose = close[i-1]
            
            # Camarilla formula
            range_val = phigh - plow
            if range_val <= 0:
                h3 = plow + 1.1 * range_val * 1.1 / 12
                l3 = plow + 1.1 * range_val * 1.1 / 12
                h4 = plow + 1.1 * range_val * 1.1 / 6
                l4 = plow + 1.1 * range_val * 1.1 / 6
            else:
                h3 = plow + 1.1 * range_val * 1.1 / 12
                l3 = plow + 1.1 * range_val * 1.1 / 12
                h4 = plow + 1.1 * range_val * 1.1 / 6
                l4 = plow + 1.1 * range_val * 1.1 / 6
        else:
            h3 = l3 = h4 = l4 = 0
        
        # Volume spike: current volume > 2.0 * 20-period average (more strict)
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > h3 and curr_open <= h3  # Break above H3
        bearish_breakout = curr_close < l3 and curr_open >= l3  # Break below L3
        
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
                # Reverse breakout or trend rejection
                elif curr_close < l3 or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close > h3 or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume
        if position == 0:
            # Long: break above H3 AND price above 1d EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below L3 AND price below 1d EMA34
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

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0