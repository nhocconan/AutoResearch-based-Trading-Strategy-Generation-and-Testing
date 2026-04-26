#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeSpike
Hypothesis: 6h Camarilla R4/S4 breakouts aligned with 1-week EMA50 trend and volume spikes (>2x 50-period MA) capture strong momentum continuation in both bull and bear markets. Uses discrete position sizing (0.0, ±0.25) and ATR-based trailing stop (2.0x) for exits. Targets 12-30 trades/year by requiring weekly trend alignment, volume confirmation, and Camarilla structure—works in ranging markets by avoiding false breakouts via tight R4/S4 levels.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1-week ATR(14) for trailing stop
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['close']).shift()).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close']).shift()).abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_1w_values = atr_1w.values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_values)
    
    # Volume spike filter: volume > 2.0 * 50-period MA on 6h
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (50)
    start_idx = max(50, 14, 50)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trend_val = ema50_1w_aligned[i]
        atr_val = atr_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1w EMA50 = uptrend, price < 1w EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Calculate Camarilla levels for previous 6h bar
        if i >= 1:
            # Use previous bar's high, low, close for today's Camarilla levels
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            rng = ph - pl
            # Camarilla R4 and S4 levels (strong breakout/continuation)
            r4 = pc + (rng * 1.1 / 2)
            s4 = pc - (rng * 1.1 / 2)
        else:
            r4 = high_val
            s4 = low_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r4
        short_breakout = close_val < s4
        
        # Entry conditions: Camarilla breakout in direction of 1w trend + volume spike
        long_entry = long_breakout and is_uptrend and vol_spike
        short_entry = short_breakout and is_downtrend and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0