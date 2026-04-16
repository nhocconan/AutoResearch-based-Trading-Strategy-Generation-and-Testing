#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band mean reversion with 4h ADX regime filter and session filter.
# Long when price touches lower BB(20,2) AND 4h ADX < 20 (strong range) AND UTC 08-20.
# Short when price touches upper BB(20,2) AND 4h ADX < 20 AND UTC 08-20.
# Uses discrete position 0.20. BB captures overextended moves, 4h ADX ensures higher timeframe is ranging (no trend),
# session filter avoids low-liquidity Asian session whipsaws. Designed for ranging markets in both bull and bear.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (UTC 08-20)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # === 1h Indicators: Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_upper_values = bb_upper.values
    bb_lower_values = bb_lower.values
    bb_middle_values = bb_middle.values
    
    # Get 4h data once before loop for regime filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = pd.Series(high_4h).diff()
    tr2 = pd.Series(low_4h).diff().abs()
    tr3 = pd.Series(close_4h).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_4h).diff()
    down_move = -pd.Series(low_4h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align 4h ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for BB)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bb_upper_values[i]) or np.isnan(bb_lower_values[i]) or
            np.isnan(adx_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bb_upper = bb_upper_values[i]
        bb_lower = bb_lower_values[i]
        bb_middle = bb_middle_values[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band or ADX rises above 25 (trend emerging)
            if price >= bb_middle or adx_val > 25:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band or ADX rises above 25
            if price <= bb_middle or adx_val > 25:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price touches lower BB AND 4h ADX < 20 (strong range)
            if price <= bb_lower and adx_val < 20:
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price touches upper BB AND 4h ADX < 20
            elif price >= bb_upper and adx_val < 20:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_BB20_2_4hADX20_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0