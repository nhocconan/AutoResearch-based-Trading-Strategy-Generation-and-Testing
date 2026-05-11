#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels (R3, S3) from the 1-day timeframe act as key support/resistance.
A breakout above R3 or below S3 with volume confirmation (volume > 1.5x 20-period average) and
trend alignment (price > 50-period EMA on 4h) signals momentum continuation.
In ranging markets (ADX < 20), fade at R3/S3 for mean reversion.
Targets 20-50 trades per year on 4h timeframe.
"""

name = "4h_Camarilla_Pivot_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 4h data for EMA and volume
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Camarilla pivot levels (using previous day's OHLC) ---
    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    prev_day_high[0] = df_1d['high'].values[0]
    prev_day_low[0] = df_1d['low'].values[0]
    prev_day_close[0] = df_1d['close'].values[0]
    
    # Camarilla calculations
    range_ = prev_day_high - prev_day_low
    camarilla_mult = 1.1 / 12  # 1.1/12 for R3/S3
    r3 = prev_day_close + range_ * camarilla_mult * 4
    s3 = prev_day_close - range_ * camarilla_mult * 4
    r4 = prev_day_close + range_ * camarilla_mult * 5
    s4 = prev_day_close - range_ * camarilla_mult * 5
    
    # Align daily levels to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # --- 1d ADX for trend/ranging regime (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 4h EMA50 for trend filter ---
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- 4h Volume average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_avg_4h[i]) or 
            np.isnan(adx_4h[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR from entry (using 4h range as proxy)
                atr_est = np.abs(high_4h[i] - low_4h[i])
                if position == 1 and close_4h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 4h average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        # Trend regime: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx_4h[i] > 25
        ranging = adx_4h[i] < 20
        
        if position == 0:
            # Look for entries
            if trending and vol_confirm:
                # Trending market: breakout continuation
                if close_4h[i] > r3_4h[i] and close_4h[i] > ema50_4h[i]:
                    signals[i] = 0.25  # long breakout above R3
                    position = 1
                    entry_price = close_4h[i]
                elif close_4h[i] < s3_4h[i] and close_4h[i] < ema50_4h[i]:
                    signals[i] = -0.25  # short breakdown below S3
                    position = -1
                    entry_price = close_4h[i]
            elif ranging and vol_confirm:
                # Ranging market: mean reversion at S3/R3
                if i > 0:
                    # Rejection at R3 (failed breakout above)
                    if close_4h[i-1] > r3_4h[i-1] and close_4h[i] < r3_4h[i]:
                        signals[i] = -0.25  # short rejection at R3
                        position = -1
                        entry_price = close_4h[i]
                    # Rejection at S3 (failed breakdown below)
                    elif close_4h[i-1] < s3_4h[i-1] and close_4h[i] > s3_4h[i]:
                        signals[i] = 0.25   # long rejection at S3
                        position = 1
                        entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if trending:
                    # In trending market, trail with EMA20 or stop at S3
                    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
                    if close_4h[i] < ema20_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S3
                    elif close_4h[i] < s3_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # ranging or weak trend
                    # In ranging market, take profit at R4 or stop at S3
                    if close_4h[i] >= r4_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S3
                    elif close_4h[i] < s3_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if trending:
                    # In trending market, trail with EMA20 or stop at R3
                    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
                    if close_4h[i] > ema20_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R3
                    elif close_4h[i] > r3_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # ranging or weak trend
                    # In ranging market, take profit at S4 or stop at R3
                    if close_4h[i] <= s4_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R3
                    elif close_4h[i] > r3_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals