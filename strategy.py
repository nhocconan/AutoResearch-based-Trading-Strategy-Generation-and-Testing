#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1_S1_Breakout_Trend_Volume
Hypothesis: Use 4h/1d for signal direction (trend via EMA crossover, trend strength via ADX) and 1h only for entry timing (breakout of R1/S1 levels with volume confirmation). This reduces trade frequency by requiring alignment of higher timeframe trend with lower timeframe breakout. Works in bull via uptrend breaks above R1, in bear via downtrend breaks below S1. Volume confirms conviction. Target: 15-30 trades/year per symbol.
"""

name = "1h_4h1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    volume_1h = prices['volume'].values
    
    # --- 4h ADX for trend strength (14-period) ---
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = -np.diff(low_4h, prepend=low_4h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di_14 = np.where(tr_14 != 0, 100 * minus_dm_14 / tr_14, 0)
    dx = np.where((plus_di_14 + minus_di_14) != 0, 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # --- 1d EMA50 for trend direction ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- 1h Camarilla Levels (based on previous day) ---
    # Calculate from previous 24h bar (shifted by 24 to avoid lookahead)
    prev_close = np.roll(close_1h, 24)
    prev_high = np.roll(high_1h, 24)
    prev_low = np.roll(low_1h, 24)
    # Fill first 24 values with first bar values to avoid lookahead
    prev_close[:24] = close_1h[0]
    prev_high[:24] = high_1h[0]
    prev_low[:24] = low_1h[0]
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # --- Volume Filter: spike above 1.5x median of last 24 periods ---
    vol_median = pd.Series(volume_1h).rolling(window=24, min_periods=12).median().values
    vol_threshold = vol_median * 1.5
    
    # --- ATR for stoploss (14-period) ---
    tr1_1h = np.abs(high_1h - low_1h)
    tr2_1h = np.abs(high_1h - np.roll(close_1h, 1))
    tr3_1h = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    tr_1h[0] = tr1_1h[0]
    atr_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 1h
    adx_14_aligned = align_htf_to_ltf(prices, df_4h, adx_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 100  # for ADX and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(atr_1h[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_1h[i] <= entry_price - 2.5 * atr_1h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1h[i] >= entry_price + 2.5 * atr_1h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        # Determine trend conditions
        # ADX > 25 indicates strong trend
        strong_trend = adx_14_aligned[i] > 25
        # 1d trend: price above/below EMA50
        uptrend = close_1h[i] > ema50_1d_aligned[i]
        downtrend = close_1h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_1h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with strong trend and volume spike
            if close_1h[i] > camarilla_r1[i] and uptrend and strong_trend and vol_ok:
                # Long: price breaks above R1 + 1d uptrend + strong trend + volume spike
                signals[i] = 0.20
                position = 1
                entry_price = close_1h[i]
            elif close_1h[i] < camarilla_s1[i] and downtrend and strong_trend and vol_ok:
                # Short: price breaks below S1 + 1d downtrend + strong trend + volume spike
                signals[i] = -0.20
                position = -1
                entry_price = close_1h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_1h[i] <= entry_price - 2.5 * atr_1h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S1
                elif close_1h[i] <= camarilla_s1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Stoploss
                if close_1h[i] >= entry_price + 2.5 * atr_1h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R1
                elif close_1h[i] >= camarilla_r1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals