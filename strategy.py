#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike_ATRstop_v4
Hypothesis: 6h Camarilla R4/S4 breakouts with 1d EMA50 trend filter and volume spike capture institutional moves. Uses discrete sizing (0.25) and ATR stop (2.0) with 6-bar minimum hold. Only trades breakouts in 1d trend direction to avoid counter-trend whipsaw. Targets 12-37 trades/year on 6h timeframe. Version 4 adds additional delay for 1d EMA50 alignment to ensure trend is based on completed daily candle, and tightens volume confirmation to 3.0x average to reduce overtrading.
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
    
    # Get 1d data for trend filter (EMA50) and ATR - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR for stoploss calculation
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 6h data for Camarilla calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for 6h based on previous bar
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r4 = prev_close + 1.5 * (prev_high - prev_low)  # R4 level
    camarilla_s4 = prev_close - 1.5 * (prev_high - prev_low)  # S4 level
    
    # Align Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s4)
    
    # Volume filter: volume > 3.0x 20-period average (tighter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (3.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    highest_since_entry = 0.0  # for trailing stop
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA50 needs 50, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        
        # Get 6h close aligned for direct comparison
        close_6h_aligned = align_htf_to_ltf(prices, df_6h, close_6h)
        close_6h_val = close_6h_aligned[i]
        is_uptrend = close_6h_val > ema_50_val
        
        if position == 0:
            # Look for entry signals: breakout in direction of 1d trend
            long_signal = (close_6h_val > r4_val) and is_uptrend and vol_spike[i]
            short_signal = (close_6h_val < s4_val) and (not is_uptrend) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_6h_val
                bars_since_entry = 0
                highest_since_entry = close_6h_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_6h_val
                bars_since_entry = 0
                highest_since_entry = close_6h_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            highest_since_entry = max(highest_since_entry, close_6h_val)
            # Exit conditions:
            # 1. Minimum holding period of 6 bars to avoid whipsaw
            # 2. ATR-based trailing stop: 2.0 * ATR below highest since entry
            if bars_since_entry >= 6:
                trailing_stop = highest_since_entry - (2.0 * atr_val)
                if close_6h_val < trailing_stop:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            highest_since_entry = min(highest_since_entry, close_6h_val)
            # Exit conditions:
            # 1. Minimum holding period of 6 bars to avoid whipsaw
            # 2. ATR-based trailing stop: 2.0 * ATR above lowest since entry
            if bars_since_entry >= 6:
                trailing_stop = highest_since_entry + (2.0 * atr_val)
                if close_6h_val > trailing_stop:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike_ATRstop_v4"
timeframe = "6h"
leverage = 1.0