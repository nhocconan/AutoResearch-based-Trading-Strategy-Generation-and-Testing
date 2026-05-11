#!/usr/bin/env python3
"""
1d_VolatilityBreakout_HTFTrend
Hypothesis: Price breaks above/below weekly pivot levels (R1/S1) with volatility expansion (weekly ATR ratio > 1.2) and volume surge signal momentum continuation. In low volatility (weekly ATR ratio < 0.8), mean reversion at pivot levels. Uses 1d timeframe with 1h ATR for precise timing. Designed to work in both bull and bear regimes via volatility-based regime switching.
"""

name = "1d_VolatilityBreakout_HTFTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for pivot calculation (weekly timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get 1h data for ATR (volatility regime)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # 1d OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- 1h ATR for volatility regime (14 period) ---
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = np.abs(high_1h - low_1h)
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR ratio: current ATR / 20-period average ATR
    atr_ma_1h = pd.Series(atr_1h).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1h / (atr_ma_1h + 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1h, atr_ratio)
    
    # --- Weekly Pivot Points (using previous week's OHLC) ---
    # Calculate from previous week's OHLC
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    
    # Pivot point calculation
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly levels to 1d
    pivot_1d = align_htf_to_ltf(prices, df_1w, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    r2_1d = align_htf_to_ltf(prices, df_1w, r2)
    s2_1d = align_htf_to_ltf(prices, df_1w, s2)
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for ATR ratio and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(pivot_1d[i]) or 
            np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(vol_avg_1d[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR from entry
                atr_est = np.abs(high_1d[i] - low_1d[i])  # rough 1d ATR estimate
                if position == 1 and close_1d[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1d[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volatility regime: high vol = breakout mode, low vol = mean reversion
        high_vol = atr_ratio_1d_aligned[i] > 1.2
        low_vol = atr_ratio_1d_aligned[i] < 0.8
        
        # Volume confirmation: current volume > 1.3x 1d average
        vol_confirm = volume_1d[i] > 1.3 * vol_avg_1d[i]
        
        if position == 0:
            # Look for entries based on volatility regime
            if high_vol and vol_confirm:
                # High volatility: breakout continuation
                if close_1d[i] > r1_1d[i]:
                    signals[i] = 0.25  # long breakout above R1
                    position = 1
                    entry_price = close_1d[i]
                elif close_1d[i] < s1_1d[i]:
                    signals[i] = -0.25  # short breakdown below S1
                    position = -1
                    entry_price = close_1d[i]
            elif low_vol and vol_confirm:
                # Low volatility: mean reversion at pivot levels
                if i > 0:
                    # Rejection at R1 (failed breakout above)
                    if close_1d[i-1] > r1_1d[i-1] and close_1d[i] < r1_1d[i]:
                        signals[i] = -0.25  # short rejection at R1
                        position = -1
                        entry_price = close_1d[i]
                    # Rejection at S1 (failed breakdown below)
                    elif close_1d[i-1] < s1_1d[i-1] and close_1d[i] > s1_1d[i]:
                        signals[i] = 0.25   # long rejection at S1
                        position = 1
                        entry_price = close_1d[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if high_vol:
                    # In high vol, trail with 1h EMA20 or stop at S1
                    ema20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema20_1h)
                    if not np.isnan(ema20_1h_aligned[i]) and close_1d[i] < ema20_1h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_1d[i] < s1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # low_vol or neutral
                    # In low vol, take profit at R2 or stop at S1
                    if close_1d[i] >= r2_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_1d[i] < s1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if high_vol:
                    # In high vol, trail with 1h EMA20 or stop at R1
                    ema20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema20_1h)
                    if not np.isnan(ema20_1h_aligned[i]) and close_1d[i] > ema20_1h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_1d[i] > r1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # low_vol or neutral
                    # In low vol, take profit at S2 or stop at R1
                    if close_1d[i] <= s2_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_1d[i] > r1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals