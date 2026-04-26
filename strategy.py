#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_ADX
Hypothesis: On 6h timeframe, Elder Ray Bull/Bear Power combined with ADX regime filter captures trend strength and momentum exhaustion. Bull Power > 0 and Bear Power < 0 with rising ADX (>25) indicates strong trend continuation. Works in both bull/bear markets by adapting to regime: ADX > 25 = trend follow, ADX < 20 = mean revert at extremes. Designed for 12-37 trades/year with discrete sizing (±0.25) and ATR-based trailing stop (2.5x) to minimize fee drag.
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
    
    # Load 1d data ONCE before loop for EMA13 (Elder Ray) and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (1d)
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX (1d) - need +DI, -DI, DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI, -DI, DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h ATR(20) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_6h_values = atr_6h.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA (13), ADX (14+14), ATR (20)
    start_idx = max(13, 28, 20) + 4  # +4 to ensure 1d bar completion (6h -> 1d: 4 bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_13_aligned[i]
        adx_val = adx_aligned[i]
        atr_val = atr_6h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema_val) or np.isnan(adx_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
        bull_power = high_val - ema_val
        bear_power = low_val - ema_val
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        adx_trending = adx_val > 25
        adx_ranging = adx_val < 20
        
        # Entry conditions
        # Trending regime: follow Elder Ray signals
        long_entry = adx_trending and (bull_power > 0) and (bear_power < 0)
        short_entry = adx_trending and (bull_power < 0) and (bear_power > 0)
        
        # Ranging regime: mean reversion at extremes
        # Long when Bear Power deeply negative and turning up
        # Short when Bull Power deeply positive and turning down
        long_entry_range = adx_ranging and (bear_power < -0.5 * atr_val) and (i > start_idx) and (bear_power > (low[i-1] - ema_13_aligned[i-1]))
        short_entry_range = adx_ranging and (bull_power > 0.5 * atr_val) and (i > start_idx) and (bull_power < (high[i-1] - ema_13_aligned[i-1]))
        
        long_entry = long_entry or long_entry_range
        short_entry = short_entry or short_entry_range
        
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
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
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

name = "6h_ElderRay_BullBearPower_Regime_ADX"
timeframe = "6h"
leverage = 1.0