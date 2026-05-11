#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Camarilla pivot levels (R1/S1) from daily data act as key support/resistance. 
When price breaks above R1 or below S1 with volume confirmation (1.5x average) and 
12h EMA50 trend alignment, it signals momentum continuation. In low volatility 
regimes (ATR ratio < 0.8), fade at R1/S1 for mean reversion. Uses 4h timeframe 
with 12h EMA trend filter and volume confirmation. Targets 20-50 trades/year.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d ATR for volatility regime (14 period) ---
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
    # ATR ratio: current ATR / 20-period average ATR
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / (atr_ma_1d + 1e-10)
    atr_ratio_4h_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # --- Daily Camarilla Pivot Levels (using previous day's OHLC) ---
    # Calculate from previous day's OHLC
    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    prev_day_high[0] = df_1d['high'].values[0]
    prev_day_low[0] = df_1d['low'].values[0]
    prev_day_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot point calculation
    close_prev = prev_day_close
    high_prev = prev_day_high
    low_prev = prev_day_low
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r1 = close_prev + (range_prev * 1.1 / 12)
    s1 = close_prev - (range_prev * 1.1 / 12)
    r2 = close_prev + (range_prev * 1.1 / 6)
    s2 = close_prev - (range_prev * 1.1 / 6)
    r3 = close_prev + (range_prev * 1.1 / 4)
    s3 = close_prev - (range_prev * 1.1 / 4)
    r4 = close_prev + (range_prev * 1.1 / 2)
    s4 = close_prev - (range_prev * 1.1 / 2)
    
    # Align daily levels to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # --- 12h EMA50 for trend filter ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- 4h Volume Average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and ATR ratio
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(atr_ratio_4h_aligned[i]) or np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Simple stoploss: 2.0x ATR from entry
                atr_est = np.abs(high_4h[i] - low_4h[i])  # rough 4h ATR estimate
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volatility regime: high vol = breakout mode, low vol = mean reversion
        high_vol = atr_ratio_4h_aligned[i] > 1.2
        low_vol = atr_ratio_4h_aligned[i] < 0.8
        
        # Volume confirmation: current volume > 1.5x 4h average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        # Trend filter: price above/below 12h EMA50
        price_vs_ema = close_4h[i] - ema50_12h_aligned[i]
        uptrend = price_vs_ema > 0
        downtrend = price_vs_ema < 0
        
        if position == 0:
            # Look for entries based on volatility regime
            if high_vol and vol_confirm:
                # High volatility: breakout continuation with trend filter
                if close_4h[i] > r1_4h[i] and uptrend:
                    signals[i] = 0.25  # long breakout above R1
                    position = 1
                    entry_price = close_4h[i]
                elif close_4h[i] < s1_4h[i] and downtrend:
                    signals[i] = -0.25  # short breakdown below S1
                    position = -1
                    entry_price = close_4h[i]
            elif low_vol and vol_confirm:
                # Low volatility: mean reversion at pivot levels
                if i > 0:
                    # Rejection at R1 (failed breakout above)
                    if close_4h[i-1] > r1_4h[i-1] and close_4h[i] < r1_4h[i]:
                        signals[i] = -0.25  # short rejection at R1
                        position = -1
                        entry_price = close_4h[i]
                    # Rejection at S1 (failed breakdown below)
                    elif close_4h[i-1] < s1_4h[i-1] and close_4h[i] > s1_4h[i]:
                        signals[i] = 0.25   # long rejection at S1
                        position = 1
                        entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if high_vol:
                    # In high vol, trail with 12h EMA50 or stop at S1
                    if close_4h[i] < ema50_12h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_4h[i] < s1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # low_vol or neutral
                    # In low vol, take profit at R2 or stop at S1
                    if close_4h[i] >= r2_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_4h[i] < s1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if high_vol:
                    # In high vol, trail with 12h EMA50 or stop at R1
                    if close_4h[i] > ema50_12h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_4h[i] > r1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # low_vol or neutral
                    # In low vol, take profit at S2 or stop at R1
                    if close_4h[i] <= s2_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_4h[i] > r1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals