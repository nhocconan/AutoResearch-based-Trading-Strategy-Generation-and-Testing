#!/usr/bin/env python3
"""
6h_PivotReversal_DailyHTF
Hypothesis: On 6h timeframe, fade extreme moves using daily pivot points (R3/S3) as dynamic support/resistance,
with trend filter from 1d EMA50 and volume confirmation. In ranging markets (BTC/ETH 2025+), price often
reverts from daily R3/S3 levels. In trending markets, breakouts beyond R4/S4 with volume continue the trend.
Uses discrete position sizing (0.25) to limit fee churn. Designed for 15-30 trades/year.
Works in both bull (trend-following breakouts) and bear (mean-reversion at extremes) via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily close for EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Daily OHLC for pivot point calculation (standard floor trader pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    PP = (high_1d + low_1d + close_1d) / 3.0
    # R1 = PP * 2 - L
    R1 = PP * 2.0 - low_1d
    # S1 = PP * 2 - H
    S1 = PP * 2.0 - high_1d
    # R2 = PP + (H - L)
    R2 = PP + (high_1d - low_1d)
    # S2 = PP - (H - L)
    S2 = PP - (high_1d - low_1d)
    # R3 = PP + 2*(H - L)
    R3 = PP + 2.0 * (high_1d - low_1d)
    # S3 = PP - 2*(H - L)
    S3 = PP - 2.0 * (high_1d - low_1d)
    # R4 = PP + 3*(H - L)  (extreme breakout level)
    R4 = PP + 3.0 * (high_1d - low_1d)
    # S4 = PP - 3*(H - L)  (extreme breakdown level)
    S4 = PP - 3.0 * (high_1d - low_1d)
    
    # Align all pivot levels to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike: current volume > 1.8 * 30-period average (6h bars = 7.5h lookback)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    # ATR for dynamic stop (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for EMA, volume avg, ATR
    start_idx = max(50, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        ema_trend = ema_1d_aligned[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entries
            # Mean reversion long: price at or below S3 with volume spike AND above daily EMA (bullish bias)
            mr_long = (close_val <= S3_aligned[i]) and volume_spike[i] and (close_val > ema_trend)
            # Mean reversion short: price at or above R3 with volume spike AND below daily EMA (bearish bias)
            mr_short = (close_val >= R3_aligned[i]) and volume_spike[i] and (close_val < ema_trend)
            # Breakout long: price breaks above R4 with volume spike AND above daily EMA
            breakout_long = (close_val > R4_aligned[i]) and volume_spike[i] and (close_val > ema_trend)
            # Breakout short: price breaks below S4 with volume spike AND below daily EMA
            breakout_short = (close_val < S4_aligned[i]) and volume_spike[i] and (close_val < ema_trend)
            
            if mr_long or breakout_long:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif mr_short or breakout_short:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            # Take profit at R2 (mean reversion) or R4 (breakout extension)
            tp_condition = (close_val >= R2_aligned[i]) or (close_val >= R4_aligned[i])
            # Stop loss: close below S3 (mean reversion failure) or ATR stop
            sl_condition = (close_val < S3_aligned[i]) or (close_val < entry_price - 2.5 * atr_val)
            if tp_condition or sl_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            # Take profit at S2 (mean reversion) or S4 (breakout extension)
            tp_condition = (close_val <= S2_aligned[i]) or (close_val <= S4_aligned[i])
            # Stop loss: close above R3 (mean reversion failure) or ATR stop
            sl_condition = (close_val > R3_aligned[i]) or (close_val > entry_price + 2.5 * atr_val)
            if tp_condition or sl_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_PivotReversal_DailyHTF"
timeframe = "6h"
leverage = 1.0