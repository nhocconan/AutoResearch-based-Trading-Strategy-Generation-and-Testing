#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_RegimeFilter_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) combined with 1d EMA50 trend filter and 6h ADX regime filter captures sustained momentum moves in both bull and bear markets. Bull Power > 0 + ADX > 25 + price > 1d EMA50 = long; Bear Power < 0 + ADX > 25 + price < 1d EMA50 = short. Discrete sizing (0.25) targets 12-30 trades/year. Works in bull/bear by taking entries only when higher-timeframe trend and momentum align with strong trend regime (ADX > 25). Avoids whipsaws in ranging markets via ADX filter.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ADX(14) for regime filter (trending vs ranging)
    # Calculate True Range
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA50 (50), ADX (14*2), EMA13 (13)
    start_idx = max(50, 28, 13)
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema50_val = ema50_1d_aligned[i]
        adx_val = adx[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Skip if any data not ready
        if (np.isnan(ema50_val) or np.isnan(adx_val) or np.isnan(bull_val) or np.isnan(bear_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        is_uptrend = close_val > ema50_val
        is_downtrend = close_val < ema50_val
        
        # Regime filter: ADX > 25 indicates trending market
        is_trending = adx_val > 25
        
        # Elder Ray signals
        bull_strong = bull_val > 0  # Bull Power positive
        bear_strong = bear_val < 0  # Bear Power negative
        
        # Entry conditions: Elder Ray in direction of 1d trend + trending regime (ADX > 25)
        long_entry = bull_strong and is_uptrend and is_trending
        short_entry = bear_strong and is_downtrend and is_trending
        
        # Exit conditions: reverse of entry or regime change to ranging
        long_exit = not (bull_strong and is_uptrend and is_trending)
        short_exit = not (bear_strong and is_downtrend and is_trending)
        
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

name = "6h_ElderRay_BullBearPower_1dTrend_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0