#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA50 Trend and Volume Spike
Hypothesis: Daily Donchian channel breakouts capture major trend moves. 
1-week EMA50 provides higher-timeframe trend filter to avoid counter-trend trades.
Volume confirmation ensures breakout validity. Works in bull markets via long 
breakouts above upper channel and in bear markets via short breakdowns below 
lower channel. ATR-based trailing stop manages risk. Target: 30-100 total trades 
over 4 years (7-25/year) on 1d timeframe.
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
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 1d
    dc_high = np.full(len(df_1d), np.nan)
    dc_low = np.full(len(df_1d), np.nan)
    
    for i in range(20, len(df_1d)):
        dc_high[i] = np.max(df_1d['high'].iloc[i-19:i+1].values)
        dc_low[i] = np.min(df_1d['low'].iloc[i-19:i+1].values)
    
    # Align Donchian levels to 1d timeframe (same timeframe, so direct use after warmup)
    dc_high_aligned = dc_high  # Already on 1d timeframe
    dc_low_aligned = dc_low    # Already on 1d timeframe
    
    # Calculate 20-period volume MA for volume confirmation (1d)
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20[i] = np.mean(df_1d['volume'].iloc[i-19:i+1].values)
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate ATR(14) for stoploss (1d)
    atr_14 = np.full(len(df_1d), np.nan)
    tr = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr[i] = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i], 
                    abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]), 
                    abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
    for i in range(14, len(df_1d)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for EMA50_1w, Donchian, volume MA, ATR to propagate
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(dc_high_aligned[i]) or 
            np.isnan(dc_low_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_1w = ema_50_1w_aligned[i]
        dc_high_val = dc_high_aligned[i]
        dc_low_val = dc_low_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        atr = atr_14_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: close above upper Donchian with volume confirmation and 1w EMA50 uptrend
            long_breakout = (curr_close > dc_high_val) and volume_confirm and (curr_close > ema50_1w)
            # Short breakdown: close below lower Donchian with volume confirmation and 1w EMA50 downtrend
            short_breakout = (curr_close < dc_low_val) and volume_confirm and (curr_close < ema50_1w)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            atr_stop = max(atr_stop, curr_high - 2.0 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            atr_stop = min(atr_stop, curr_low + 2.0 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0