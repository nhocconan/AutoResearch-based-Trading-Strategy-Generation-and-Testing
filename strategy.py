#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeADX
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts aligned with 1d EMA34 trend, volume spike (>1.8x 20-period MA), and ADX regime filter (ADX>25 for trending) capture high-probability continuation moves while avoiding chop. Uses discrete position sizing (0.0, ±0.25) and ATR-based trailing stop (2.5x) for exits. Targets 20-40 trades/year by requiring multiple confluence factors—designed to work in bull (trend continuation up) and bear (trend continuation down) markets.
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
    
    # 1d ATR(14) for trailing stop
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_1d_values = atr_1d.values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_values)
    
    # 1d ADX(14) for regime filter
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr_14 = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / tr_14)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / tr_14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume spike filter: volume > 1.8 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA34 (34), ATR (14), ADX (14), volume MA (20)
    start_idx = max(34, 14, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trend_val = ema34_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(adx_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA34 = uptrend, price < 1d EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Regime filter: ADX > 25 indicates trending market (avoid chop)
        is_trending = adx_val > 25
        
        # Calculate Camarilla levels for previous 4h bar
        if i >= 1:
            # Use previous bar's high, low, close for today's Camarilla levels
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            rng = ph - pl
            # Camarilla R1 and S1 levels (primary breakout)
            r1 = pc + (rng * 1.1 / 12)  # R1 = C + (H-L)*1.1/12
            s1 = pc - (rng * 1.1 / 12)  # S1 = C - (H-L)*1.1/12
        else:
            r1 = high_val
            s1 = low_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r1
        short_breakout = close_val < s1
        
        # Entry conditions: Camarilla breakout in direction of 1d trend + volume spike + trending regime
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeADX"
timeframe = "4h"
leverage = 1.0