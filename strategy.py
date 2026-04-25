#!/usr/bin/env python3
"""
6h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels from 1d act as strong support/resistance on 6h.
Breakout through these levels with 1d EMA34 trend alignment and volume confirmation
captures institutional flow while avoiding overtrading. Uses discrete position sizing (0.25) and volume threshold (2.0x)
to target 50-150 total trades over 4 years (12-37/year). Works in bull/bear by following 1d trend while using Camarilla levels
for precise entry/exit. Added ATR-based trailing stop to reduce drawdown and improve Sharpe.
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
    
    # Calculate Camarilla levels from previous day (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 6h timeframe
    prev_high_6h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_6h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_6h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels: H3/L3
    rng = prev_high_6h - prev_low_6h
    h3 = prev_close_6h + rng * 1.1 / 6.0
    l3 = prev_close_6h - rng * 1.1 / 6.0
    
    # 1d EMA34 for trend filter (more stable than 12h)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Camarilla (1d) + EMA34 (1d) + VolMA20 + ATR
    start_idx = max(50, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_level = ema_34_1d_aligned[i]
        atr_value = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average (reduced from 2.5x for more trades)
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        bullish_breakout = curr_close > h3[i]  # Break above H3
        bearish_breakout = curr_close < l3[i]  # Break below L3
        
        # Update tracking variables for trailing stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout or trend change
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.0*ATR from highest since entry
                if curr_close < highest_since_entry - 2.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend change
                elif curr_close < l3[i] or curr_close < ema_34_level:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend change
                elif curr_close > h3[i] or curr_close > ema_34_level:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_price = 0.0
                continue
        
        # Entry conditions: Breakout + trend + volume
        if position == 0:
            long_condition = bullish_breakout and (curr_close > ema_34_level) and volume_spike
            short_condition = bearish_breakout and (curr_close < ema_34_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0