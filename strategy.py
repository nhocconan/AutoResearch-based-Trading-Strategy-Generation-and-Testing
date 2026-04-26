#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike
Hypothesis: On 6h timeframe, Camarilla R4/S4 breakouts aligned with 12h EMA50 trend and volume spikes (>1.8x 20-period MA) capture high-probability continuation moves. R4/S4 levels represent stronger breakout points than R1/S1, reducing false signals. Uses discrete position sizing (0.0, ±0.25) and 12h ATR-based trailing stop (2.0x) for exits. Targets 12-25 trades/year by requiring HTF trend alignment, volume confirmation, and Camarilla structure—designed to work in both bull (trend continuation) and bear (trend continuation down) markets.
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
    
    # Load 12h data ONCE before loop for HTF filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h ATR(14) for trailing stop
    tr1 = pd.Series(df_12h['high']).diff().abs()
    tr2 = (pd.Series(df_12h['high']) - pd.Series(df_12h['close']).shift()).abs()
    tr3 = (pd.Series(df_12h['low']) - pd.Series(df_12h['close']).shift()).abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_12h_values = atr_12h.values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h_values)
    
    # Volume spike filter: volume > 1.8 * 20-period MA on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trend_val = ema50_12h_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 12h EMA50 = uptrend, price < 12h EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Calculate Camarilla levels for previous 6h bar
        if i >= 1:
            # Use previous bar's high, low, close for today's Camarilla levels
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            rng = ph - pl
            # Camarilla R4 and S4 levels (stronger breakout)
            r4 = pc + (rng * 1.1 / 2)  # R4 = C + (H-L)*1.1/2
            s4 = pc - (rng * 1.1 / 2)  # S4 = C - (H-L)*1.1/2
        else:
            r4 = high_val
            s4 = low_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r4
        short_breakout = close_val < s4
        
        # Entry conditions: Camarilla breakout in direction of 12h trend + volume spike
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

name = "6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0