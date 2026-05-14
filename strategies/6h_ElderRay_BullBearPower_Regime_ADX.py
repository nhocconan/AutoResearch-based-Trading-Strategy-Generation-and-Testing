#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_ADX
Hypothesis: On 6h timeframe, Elder Ray Bull Power (high - EMA13) and Bear Power (low - EMA13) combined with ADX regime filter captures sustainable trends while avoiding whipsaws. Long when Bull Power > 0, ADX > 25, and +DI > -DI; Short when Bear Power < 0, ADX > 25, and -DI > +DI. Uses 1d EMA50 as higher-timeframe trend filter to avoid counter-trend trades. Designed for 12-37 trades/year with discrete sizing (±0.25) and close-based stops to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h)
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: high - EMA
    bear_power = low - ema_13   # Bear Power: low - EMA
    
    # ADX calculation (6h)
    # True Range
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high).diff()
    down_move = -(pd.Series(low).diff())  # negative of low diff
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA13 (13), ADX (14*2=28 for smoothing), 1d EMA50 alignment
    start_idx = max(13, 28) + 4  # +4 to ensure 1d bar completion (6h -> 1d: 4 bars per day)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        adx_val = adx[i]
        pdi = plus_di[i]
        mdi = minus_di[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        
        # Regime conditions: ADX > 25 for trending market
        strong_trend = adx_val > 25
        
        # Entry conditions
        long_entry = (bp > 0) and strong_trend and (pdi > mdi) and (close_val > ema_50_val)
        short_entry = (br < 0) and strong_trend and (mdi > pdi) and (close_val < ema_50_val)
        
        # Exit conditions: reverse signal or trend deterioration
        long_exit = (bp <= 0) or (adx_val < 20) or (mdi > pdi) or (close_val < ema_50_val)
        short_exit = (br >= 0) or (adx_val < 20) or (pdi > mdi) or (close_val > ema_50_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_ADX"
timeframe = "6h"
leverage = 1.0