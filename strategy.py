#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 12h ADX regime filter.
- Primary timeframe: 6h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ADX for regime detection (ADX > 25 = trending, ADX < 20 = ranging).
- Elder Ray: Bull Power = High - EMA13(Close), Bear Power = EMA13(Close) - Low.
- Entry: Long when Bull Power > 0 AND ADX > 25 (strong uptrend).
         Short when Bear Power > 0 AND ADX > 25 (strong downtrend).
- Exit: When ADX < 20 (regime shift to ranging) or power signals reverse.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures trending moves with institutional buying/selling pressure,
avoiding ranging markets where Elder Ray gives false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 6h data for Elder Ray calculation (EMA13 of close)
    ema_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13(Close)
    bull_power = high - ema_6h
    # Bear Power = EMA13(Close) - Low
    bear_power = ema_6h - low
    
    # Align HTF ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30)  # Need enough bars for ADX and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_6h[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals in trending regime (ADX > 25)
            if adx_aligned[i] > 25:
                # Long: Bull Power > 0 (buying pressure) in strong uptrend
                if bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 (selling pressure) in strong downtrend
                elif bear_power[i] > 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long when ADX < 20 (ranging) or Bull Power <= 0 (pressure gone)
            if adx_aligned[i] < 20 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when ADX < 20 (ranging) or Bear Power <= 0 (pressure gone)
            if adx_aligned[i] < 20 or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_12hADX_Regime_v2"
timeframe = "6h"
leverage = 1.0