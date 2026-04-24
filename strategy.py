#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w volume spike and 1w ADX regime filter.
- Primary timeframe: 1d for execution and Camarilla pivots, HTF: 1w for volume spike and ADX.
- ADX > 25 indicates trending market (breakout strategy), ADX < 20 indicates ranging (mean reversion at H3/L3).
- Entry: Long when price breaks above H3 AND ADX > 25 AND volume spike.
         Short when price breaks below L3 AND ADX > 25 AND volume spike.
         In ranging (ADX < 20): Long when price touches L3 AND reverses up (close > low).
                                Short when price touches H3 AND reverses down (close < high).
- Exit: Opposite Camarilla breakout or ADX regime shift to ranging.
- Volume confirmation: current 1w volume > 1.5 * 20-period 1w volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) on 1d
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    # Camarilla width = (H - L) * 1.1 / 8
    width = (df_1d['high'] - df_1d['low']) * 1.1 / 8.0
    # H3 = C + width * 1.1
    camarilla_h3 = df_1d['close'].values + width * 1.1
    # L3 = C - width * 1.1
    camarilla_l3 = df_1d['close'].values - width * 1.1
    
    # Get 1w data for ADX and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1w
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current 1w volume > 1.5 * 20-period 1w volume MA
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (1.5 * volume_ma_1w)
    
    # Align HTF indicators to 1d
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1w bars for ADX/volume MA and 30 for 1d pivots
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                if adx_val > 25:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above H3
                    if curr_close > h3:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below L3
                    elif curr_close < l3:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at extremes
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3 and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3 and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR ADX drops to ranging
            if curr_close < l3 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR ADX drops to ranging
            if curr_close > h3 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_1wVolumeSpike_1wADXRegime_v1"
timeframe = "1d"
leverage = 1.0