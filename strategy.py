#!/usr/bin/env python3
"""
6h_KeltnerChannel_Breakout_1dTrend_ADXFilter
Hypothesis: Combines Keltner Channel breakouts with 1d EMA50 trend filter and ADX(14) > 25 for trend strength.
Enters long when price breaks above upper KC AND 1d close > EMA50 AND ADX > 25.
Enters short when price breaks below lower KC AND 1d close < EMA50 AND ADX > 25.
Exits when price returns to middle line (EMA20 of typical price) OR trend weakens (ADX < 20).
Designed for 6h timeframe to achieve 50-150 total trades over 4 years with controlled risk.
Works in both bull and bear markets by requiring strong trend (ADX>25) and following 1d trend direction.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Keltner Channel (20, ATR=10) on 6h data
    typical_price = (high + low + close) / 3
    tp_series = pd.Series(typical_price)
    ema_20_tp = tp_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_series = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    kc_upper = ema_20_tp + (2 * atr_series)
    kc_lower = ema_20_tp - (2 * atr_series)
    kc_middle = ema_20_tp  # Middle line is EMA20 of typical price
    
    # ADX(14) for trend strength filter
    # +DM, -DM, TR
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed values
    tr_series = pd.Series(tr)
    atr_14 = tr_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    plus_di_14 = 100 * (plus_dm_series.ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14)
    minus_di_14 = 100 * (minus_dm_series.ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA20(20), ATR(10), ADX(14+14=28), 1d EMA50(50)
    start_idx = max(20, 10, 28, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(kc_middle[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        kc_up = kc_upper[i]
        kc_low = kc_lower[i]
        kc_mid = kc_middle[i]
        adx_val = adx[i]
        
        if position == 0:
            # Look for entry: breakout of KC with 1d trend filter AND strong trend (ADX>25)
            # Long: price breaks above upper KC AND 1d uptrend AND ADX > 25
            long_condition = (close_val > kc_up) and (close_val > ema_val) and (adx_val > 25)
            # Short: price breaks below lower KC AND 1d downtrend AND ADX > 25
            short_condition = (close_val < kc_low) and (close_val < ema_val) and (adx_val > 25)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to middle line OR trend weakens (ADX<20) OR trend breaks
            exit_condition = (close_val <= kc_mid) or (adx_val < 20) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to middle line OR trend weakens (ADX<20) OR trend breaks
            exit_condition = (close_val >= kc_mid) or (adx_val < 20) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_KeltnerChannel_Breakout_1dTrend_ADXFilter"
timeframe = "6h"
leverage = 1.0