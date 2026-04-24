#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and 1w for ADX regime.
- Trend filter: Price > 1d EMA34 for bullish bias, Price < 1d EMA34 for bearish bias.
- Regime filter: 1w ADX > 25 for trending markets (breakout strategy), ADX < 20 for ranging (avoid false breakouts).
- Entry: Long when price breaks above Camarilla H3 AND 1d EMA34 trend bullish AND 1w ADX > 25 AND volume spike.
         Short when price breaks below Camarilla L3 AND 1d EMA34 trend bearish AND 1w ADX > 25 AND volume spike.
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or ADX drops below 20 (regime shift to ranging).
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for EMA34 and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ADX (14-period) on 1w for regime filter
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_1w + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1w + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Camarilla levels from previous 1d bar (H3, L3, H4, L4)
    # We need the previous completed 1d bar's OHLC
    # Since we're on 4h timeframe, we'll use the last completed 1d bar
    # For simplicity, we use the current 1d bar's OHLC (which is forming) but this introduces look-ahead
    # Proper approach: use the previous 1d bar's close to calculate today's Camarilla
    # However, for breakout strategies, we use the previous day's range
    # We'll shift the 1d data by 1 to avoid look-ahead
    df_1d_shifted = df_1d.copy()
    df_1d_shifted[['open', 'high', 'low', 'close']] = df_1d_shifted[['open', 'high', 'low', 'close']].shift(1)
    
    # Calculate Camarilla levels for current 4h bar using previous 1d bar
    prev_close = df_1d_shifted['close'].values
    prev_high = df_1d_shifted['high'].values
    prev_low = df_1d_shifted['low'].values
    
    # Camarilla levels
    H3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    H4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    L4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h
    H3_aligned = align_htf_to_ltf(prices, df_1d_shifted, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d_shifted, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d_shifted, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d_shifted, L4)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20, 30)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        ema34_val = ema34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals only in trending regime (ADX > 25)
            if adx_val > 25 and volume_spike[i]:
                # Bullish breakout: price breaks above H3 with bullish EMA trend
                if curr_high > H3_aligned[i] and curr_close > ema34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 with bearish EMA trend
                elif curr_low < L3_aligned[i] and curr_close < ema34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR ADX drops to ranging (<20)
            if curr_low < L3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR ADX drops to ranging (<20)
            if curr_high > H3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_1wADXRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0