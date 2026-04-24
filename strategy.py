#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ADX trend strength.
- ADX > 25 indicates trending market (breakout strategy), ADX < 20 indicates ranging (mean reversion at Camarilla H3/L3).
- Entry: Long when price breaks above Camarilla H3 AND ADX > 25 (bullish breakout in trend).
         Short when price breaks below Camarilla L3 AND ADX > 25 (bearish breakout in trend).
         In ranging (ADX < 20): Long when price touches Camarilla L3 AND reverses up (close > low).
                                Short when price touches Camarilla H3 AND reverses down (close < high).
- Exit: Opposite Camarilla breakout or ADX regime shift to ranging.
- Volume confirmation: current volume > 1.3 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Camarilla levels from previous 1d
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    prev_close = pd.Series(df_1d['close']).shift(1).values
    prev_high = pd.Series(df_1d['high']).shift(1).values
    prev_low = pd.Series(df_1d['low']).shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2.0
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2.0
    
    # Align Camarilla levels to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for ADX and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above Camarilla H3
                    if curr_close > camarilla_h3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below Camarilla L3
                    elif curr_close < camarilla_l3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at extremes
                    # Long when price touches lower Camarilla L3 and shows reversal (close > low)
                    if curr_low <= camarilla_l3_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Camarilla H3 and shows reversal (close < high)
                    elif curr_high >= camarilla_h3_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla H3/L3 midpoint OR ADX drops to ranging
            camarilla_mid = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0
            if curr_close < camarilla_mid or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla H3/L3 midpoint OR ADX drops to ranging
            camarilla_mid = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0
            if curr_close > camarilla_mid or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dADXRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0