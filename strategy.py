#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_ADX_Filter
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts aligned with 1d EMA34 trend and ADX > 25 capture strong continuation moves while filtering choppy markets. Uses discrete position sizing (0.0, ±0.25) and 12h ATR-based trailing stop (2.5x) for exits. Targets 12-37 trades/year by requiring HTF trend alignment, ADX regime filter, and Camarilla structure—designed to work in both bull (trend continuation) and bear (trend continuation down) markets by following the 1d EMA34 direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d ADX(14) for regime filter
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Calculate +DM and -DM
    up_move = pd.Series(df_1d['high']).diff()
    down_move = pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smooth TR, +DM, -DM
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # 12h ATR(14) for trailing stop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    tr1_12h = pd.Series(df_12h['high']).diff().abs()
    tr2_12h = (pd.Series(df_12h['high']) - pd.Series(df_12h['close']).shift()).abs()
    tr3_12h = (pd.Series(df_12h['low']) - pd.Series(df_12h['close']).shift()).abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean()
    atr_12h_values = atr_12h.values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h_values)
    
    # Volume spike filter: volume > 2.0 * 20-period MA on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA34 (34), ADX (14+14), ATR (14), volume MA (20)
    start_idx = max(34, 28, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trend_val = ema34_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(adx_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA34 = uptrend, price < 1d EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        # Regime filter: ADX > 25 indicates trending market
        is_trending = adx_val > 25
        
        # Calculate Camarilla levels for previous 12h bar
        if i >= 1:
            # Use previous bar's high, low, close for today's Camarilla levels
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            rng = ph - pl
            # Camarilla R1 and S1 levels
            r1 = pc + (rng * 1.1 / 12)  # R1 = C + (H-L)*1.1/12
            s1 = pc - (rng * 1.1 / 12)  # S1 = C - (H-L)*1.1/12
        else:
            r1 = high_val
            s1 = low_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r1
        short_breakout = close_val < s1
        
        # Entry conditions: Camarilla breakout in direction of 1d trend + volume spike + ADX > 25
        long_entry = long_breakout and is_uptrend and vol_spike and is_trending
        short_entry = short_breakout and is_downtrend and vol_spike and is_trending
        
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
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
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

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_ADX_Filter"
timeframe = "12h"
leverage = 1.0