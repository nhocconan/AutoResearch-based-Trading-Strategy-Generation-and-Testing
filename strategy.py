#!/usr/bin/env python3
"""
EXPERIMENT #021 - Keltner Breakout with ADX Trend Strength
==========================================================
Hypothesis: Keltner Channel breakouts filtered by ADX trend strength will capture 
strong momentum moves while avoiding choppy market whipsaws. 6h KAMA provides 
adaptive trend filter that adjusts to volatility regimes better than fixed EMA/HMA.

Key features:
- 6h KAMA for adaptive trend direction (Kaufman Adaptive MA adjusts to noise)
- 1h Keltner Channel (EMA20 + 1.5*ATR) breakout entries
- ADX(14) > 20 filter to ensure sufficient trend strength
- Volume confirmation (above 20-period average)
- ATR trailing stop (2.5*ATR) with position exit on stop hit
- Discrete position sizing (0.0, ±0.25, ±0.35) to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_6h_keltner_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth using Wilder's method (EMA with alpha = 1/period)
    avg_tr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_plus_dm = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_minus_dm = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = avg_tr > 0
    plus_di[mask] = 100 * avg_plus_dm[mask] / avg_tr[mask]
    minus_di[mask] = 100 * avg_minus_dm[mask] / avg_tr[mask]
    
    # Calculate DX
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask = di_sum > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i-period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period+1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # =========================================================================
    # LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL - Rule 1)
    # =========================================================================
    df_6h = get_htf_data(prices, '6h')
    kama_6h_raw = calculate_kama(df_6h['close'].values, period=10)
    kama_6h_aligned = align_htf_to_ltf(prices, df_6h, kama_6h_raw)
    
    # =========================================================================
    # CALCULATE 1H INDICATORS (vectorized before loop)
    # =========================================================================
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Keltner Channel
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 1.5 * atr_14
    kc_lower = ema_20 - 1.5 * atr_14
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # =========================================================================
    # POSITION SIZING PARAMETERS (Rule 4 - discrete levels)
    # =========================================================================
    BASE_SIZE = 0.25  # 25% of capital
    MAX_SIZE = 0.35
    STOP_MULT = 2.5   # 2.5 * ATR stop distance
    
    # =========================================================================
    # GENERATE SIGNALS
    # =========================================================================
    signals = np.zeros(n)
    entry_price = np.zeros(n)
    position_side = np.zeros(n, dtype=int)
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    # Start after all indicators are valid
    start_idx = max(50, period if 'period' in dir() else 50)
    
    for i in range(start_idx, n):
        # Skip if any indicator is NaN
        if np.isnan(kama_6h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(adx_14[i]):
            continue
        
        curr_atr = atr_14[i]
        if curr_atr <= 0:
            curr_atr = atr_14[i-1] if i > 0 else 0.001 * close[i]
        
        stop_distance = STOP_MULT * curr_atr
        
        # ---------------------------------------------------------------------
        # MANAGE EXISTING LONG POSITION
        # ---------------------------------------------------------------------
        if position_side[i-1] == 1:
            entry = entry_price[i-1]
            
            # Stop loss check
            if close[i] < entry - stop_distance:
                signals[i] = 0.0
                position_side[i] = 0
                continue
            
            # Trail stop using highest high
            if i == start_idx or highest_high[i-1] == 0:
                highest_high[i] = high[i]
            else:
                highest_high[i] = max(highest_high[i-1], high[i])
            
            # Check trailed stop
            trailed_stop = highest_high[i] - stop_distance
            if close[i] < trailed_stop and i > start_idx:
                signals[i] = 0.0
                position_side[i] = 0
                continue
            
            # Maintain position
            signals[i] = BASE_SIZE
            entry_price[i] = entry
            position_side[i] = 1
            continue
        
        # ---------------------------------------------------------------------
        # MANAGE EXISTING SHORT POSITION
        # ---------------------------------------------------------------------
        if position_side[i-1] == -1:
            entry = entry_price[i-1]
            
            # Stop loss check
            if close[i] > entry + stop_distance:
                signals[i] = 0.0
                position_side[i] = 0
                continue
            
            # Trail stop using lowest low
            if i == start_idx or lowest_low[i-1] == 0:
                lowest_low[i] = low[i]
            else:
                lowest_low[i] = min(lowest_low[i-1], low[i])
            
            # Check trailed stop
            trailed_stop = lowest_low[i] + stop_distance
            if close[i] > trailed_stop and i > start_idx:
                signals[i] = 0.0
                position_side[i] = 0
                continue
            
            # Maintain position
            signals[i] = -BASE_SIZE
            entry_price[i] = entry
            position_side[i] = -1
            continue
        
        # ---------------------------------------------------------------------
        # CHECK FOR NEW LONG ENTRY
        # ---------------------------------------------------------------------
        # Conditions: 6h KAMA bullish + ADX strong + Keltner breakout + volume
        kama_bullish = close[i] > kama_6h_aligned[i]
        adx_strong = adx_14[i] > 20
        kc_breakout_long = close[i] > kc_upper[i]
        volume_confirmed = volume[i] > 1.2 * vol_avg[i]
        
        if kama_bullish and adx_strong and kc_breakout_long and volume_confirmed:
            signals[i] = BASE_SIZE
            entry_price[i] = close[i]
            position_side[i] = 1
            highest_high[i] = high[i]
            continue
        
        # ---------------------------------------------------------------------
        # CHECK FOR NEW SHORT ENTRY
        # ---------------------------------------------------------------------
        # Conditions: 6h KAMA bearish + ADX strong + Keltner breakout + volume
        kama_bearish = close[i] < kama_6h_aligned[i]
        kc_breakout_short = close[i] < kc_lower[i]
        
        if kama_bearish and adx_strong and kc_breakout_short and volume_confirmed:
            signals[i] = -BASE_SIZE
            entry_price[i] = close[i]
            position_side[i] = -1
            lowest_low[i] = low[i]
            continue
        
        # No signal - stay flat
        signals[i] = 0.0
        position_side[i] = 0
    
    return signals