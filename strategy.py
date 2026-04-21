#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Weekly pivot points provide significant support/resistance levels. Breakout above weekly R1 with 1d EMA50 trend filter and volume confirmation captures institutional interest. Works in both bull/bear markets by requiring alignment with higher timeframe trend and volume validation, reducing false breakouts. Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === Weekly pivot points (using prior week's OHLC) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    # S1 = 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === ATR for dynamic stoploss (14-period on 6h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_50_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1_weekly = r1_1w_aligned[i]
        s1_weekly = s1_1w_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 + volume spike > 2.0 + price above 1d EMA50
            if price_close > r1_weekly and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below weekly S1 + volume spike > 2.0 + price below 1d EMA50
            elif price_close < s1_weekly and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Trailing stop: 2.0 * ATR from entry price
            if position == 1:
                if price_close < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0