#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout with 1d Volume Spike and 4h EMA34 Trend Filter
Hypothesis: Camarilla R1/S1 breakouts on 6h with daily volume spike and 4h EMA34 trend alignment capture swing moves in both bull/bear markets. Uses tighter volume confirmation (2.0x) and discrete position sizing (0.25) to target ~50-100 trades/year, minimizing fee drag while maintaining edge across regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average (tighter for fewer, higher quality trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 4h EMA34 trend filter (MTF) - loaded ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d Camarilla pivot levels (MTF) - R1/S1 for breakout, R4/S4 for stop (optional enhancement)
    camarilla_r1 = df_4h['close'] + (df_4h['high'] - df_4h['low']) * 1.1 / 12  # Using 4h for more frequent pivot updates
    camarilla_s1 = df_4h['close'] - (df_4h['high'] - df_4h['low']) * 1.1 / 12
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1.values)
    
    # Optional: Weekly pivot bias (MTF) - use 1w for longer-term direction
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bias = np.where(weekly_close > weekly_open, 1, -1)  # 1 = bullish week, -1 = bearish week
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 34) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        weekly_dir = weekly_bias_aligned[i]
        
        # Breakout conditions: price breaks Camarilla R1/S1 levels
        breakout_long = curr_close > r1_aligned[i]
        breakout_short = curr_close < s1_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: R1/S1 breakout + volume spike + 4h EMA34 trend alignment
            # Optional: weekly bias filter (only take longs in bullish week, shorts in bearish week)
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_4h_aligned[i]) and (weekly_dir >= 0)
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_4h_aligned[i]) and (weekly_dir <= 0)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike_1wBias"
timeframe = "6h"
leverage = 1.0