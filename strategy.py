#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot (H3/L3) breakout with 4h volume spike and 1d ADX regime filter.
- Primary timeframe: 1h for execution, HTF: 4h for trend confirmation, 1d for ADX trend strength.
- ADX > 25 on 1d indicates trending market (breakout strategy), ADX < 20 indicates ranging (avoid breakouts).
- Entry: Long when price breaks above Camarilla H3 AND 4h volume > 1.5 * 20-period volume MA AND ADX > 25.
         Short when price breaks below Camarilla L3 AND 4h volume > 1.5 * 20-period volume MA AND ADX > 25.
- Exit: Opposite Camarilla breakout (H3 for shorts, L3 for longs) or ADX drops below 20 (regime shift to ranging).
- Volume confirmation: 4h volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Session filter: 08-20 UTC to avoid low-volume Asian session noise.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
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
    
    # Get 4h data for Camarilla pivots and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivots (H3, L3) on 4h using previous day's OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using 4h bar's OHLC for simplicity (standard Camarilla uses daily)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    rng = high_4h - low_4h
    H3 = close_4h + 1.1 * rng / 2.0
    L3 = close_4h - 1.1 * rng / 2.0
    
    # Align Camarilla levels to 1h
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    
    # 4h volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_4h = df_4h['volume'].values
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * volume_ma_4h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
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
    
    # Align 1d ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute session filter (08-20 UTC)
    # prices.index is DatetimeIndex, .hour works directly
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough bars for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals in trending regime (ADX > 25)
            if adx_val > 25 and volume_spike_aligned[i]:
                # Bullish breakout: price breaks above H3
                if curr_high > H3_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below L3
                elif curr_low < L3_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR ADX drops to ranging (<20)
            if curr_low < L3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR ADX drops to ranging (<20)
            if curr_high > H3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_CamarillaH3L3_4hVolSpike_1dADXRegime_v1"
timeframe = "1h"
leverage = 1.0