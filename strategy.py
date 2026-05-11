#!/usr/bin/env python3
"""
4h_1d_TRIX_Volume_Spike_Regime
Hypothesis: TRIX (triple smoothed EMA) crossover with volume spike and chop regime filter.
- Long when: TRIX crosses above signal line, volume > 1.5x 20-period avg, CHOP > 61.8 (range)
- Short when: TRIX crosses below signal line, volume > 1.5x 20-period avg, CHOP > 61.8 (range)
- Exit when: TRIX crosses back through signal line
- Uses 1d trend filter to avoid counter-trend trades: only long when price > 1d EMA50, short when price < 1d EMA50
- Targets 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
- TRIX catches momentum in choppy markets; volume confirms conviction; chop filter avoids whipsaws in strong trends.
"""

name = "4h_1d_TRIX_Volume_Spike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- TRIX Indicator (12-period) ---
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then % change
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = pd.Series(ema3).pct_change() * 100  # percentage change
    trix = trix_raw.fillna(0).values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # --- Chop Index (14-period) for regime filter ---
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = log(sum(tr,14) / (highest high - lowest low,14)) * 100 / log(14)
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_hl = highest_high - lowest_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = np.log(sum_tr / range_hl) * 100 / np.log(14)
    
    # --- Volume Confirmation: 4h volume > 1.5x 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (need TRIX + chop + vol + trend)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        trix_cross_up = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
        trix_cross_down = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
        
        vol_ok = volume_4h[i] > vol_ma_20[i] * 1.5
        chop_high = chop[i] > 61.8  # ranging market
        
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Look for entries only with volume spike, in chop, and with 1d trend filter
            if trix_cross_up and vol_ok and chop_high and trend_up:
                # Long: TRIX bullish cross + volume spike + ranging + 1d uptrend
                signals[i] = 0.25
                position = 1
            elif trix_cross_down and vol_ok and chop_high and trend_down:
                # Short: TRIX bearish cross + volume spike + ranging + 1d downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit when TRIX crosses back through signal line
            if position == 1:
                if trix_cross_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if trix_cross_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals