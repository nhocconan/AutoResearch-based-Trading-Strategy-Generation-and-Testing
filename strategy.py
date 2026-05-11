#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1, S1) calculated from daily OHLC act as key support/resistance on 12h chart.
When price breaks above R1 or below S1 with 1d trend alignment (price > EMA50) and volume confirmation (>1.5x 20-period average),
it signals momentum continuation. In ranging markets (ADX < 20), fade at R1/S1 for mean reversion.
Uses 12h timeframe with 1d Camarilla levels, 1d EMA50 for trend, and ADX for regime filtering.
Targets 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels, EMA50, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Camarilla Levels (using previous day's OHLC) ---
    # Calculate from previous day's OHLC
    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    # Set first values to avoid NaN
    prev_day_high[0] = df_1d['high'].values[0]
    prev_day_low[0] = df_1d['low'].values[0]
    prev_day_close[0] = df_1d['close'].values[0]
    
    # Camarilla calculation
    range_prev = prev_day_high - prev_day_low
    camarilla_mult = 1.1 / 12  # 1.1/12 for R1/S1
    r1 = prev_day_close + range_prev * camarilla_mult
    s1 = prev_day_close - range_prev * camarilla_mult
    r2 = prev_day_close + range_prev * 1.1 / 6   # 1.1/6 for R2/S2
    s2 = prev_day_close - range_prev * 1.1 / 6
    
    # Align daily levels to 12h
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # --- 1d EMA50 for trend filter ---
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 1d ADX for regime filtering (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # --- 12h Volume Average for confirmation ---
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(adx_12h[i]) or 
            np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Hold position until clear exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_12h[i] > 1.5 * vol_avg_12h[i]
        
        # Trend filter: price above/below EMA50
        uptrend = close_12h[i] > ema50_12h[i]
        downtrend = close_12h[i] < ema50_12h[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx_12h[i] > 25
        ranging = adx_12h[i] < 20
        
        if position == 0:
            # Look for entries
            if trending and vol_confirm:
                # Trending market: breakout continuation
                if uptrend and close_12h[i] > r1_12h[i]:
                    signals[i] = 0.25  # long breakout above R1
                    position = 1
                    entry_price = close_12h[i]
                elif downtrend and close_12h[i] < s1_12h[i]:
                    signals[i] = -0.25  # short breakdown below S1
                    position = -1
                    entry_price = close_12h[i]
            elif ranging and vol_confirm:
                # Ranging market: mean reversion at S1/R1
                if close_12h[i] < s1_12h[i] and i > 0 and close_12h[i-1] >= s1_12h[i-1]:
                    signals[i] = 0.25  # long mean reversion from S1
                    position = 1
                    entry_price = close_12h[i]
                elif close_12h[i] > r1_12h[i] and i > 0 and close_12h[i-1] <= r1_12h[i-1]:
                    signals[i] = -0.25  # short mean reversion from R1
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if trending:
                    # In trending market, trail with EMA50 or stop at S1
                    if close_12h[i] < ema50_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_12h[i] < s1_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # ranging or weak trend
                    # In ranging market, take profit at R2 or stop at S1
                    if close_12h[i] >= r2_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_12h[i] < s1_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if trending:
                    # In trending market, trail with EMA50 or stop at R1
                    if close_12h[i] > ema50_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_12h[i] > r1_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # ranging or weak trend
                    # In ranging market, take profit at S2 or stop at R1
                    if close_12h[i] <= s2_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_12h[i] > r1_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals