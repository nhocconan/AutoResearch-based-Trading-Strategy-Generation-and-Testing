#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX with volume spike and choppiness regime filter for ETH/BTC.
- Primary timeframe: 12h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for TRIX calculation and 1w for choppiness regime.
- TRIX(12): Triple EMA momentum oscillator. Long when TRIX crosses above zero AND rising.
         Short when TRIX crosses below zero AND falling.
- Volume confirmation: Current volume > 2.0 * volume MA(50) on 12h.
- Choppiness regime: Only trade when CHOP(14) on 1w < 38.2 (trending market).
- Signal size: 0.25 discrete to minimize fee churn.
- ATR-based trailing stop: Exit when price moves against position by 2.5 * ATR(20).
This strategy captures strong momentum moves in trending markets with volume confirmation,
avoiding choppy regimes where TRIX whipsaws. Works in both bull and bear markets by
only taking trades in the direction of TRIX momentum with proper regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for choppiness regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate TRIX(12) on 1d: Triple EMA of close, then 1-period ROC
    df_1d_close = df_1d['close'].values
    ema1 = pd.Series(df_1d_close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (ema3_today - ema3_yesterday) / ema3_yesterday
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix_raw[0] = 0.0  # first value undefined
    
    # Calculate 1w Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high)-min(low))))
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(df_1w_high[1:] - df_1w_low[:-1])
    tr2 = np.abs(df_1w_high[1:] - df_1w_close[:-1])
    tr3 = np.abs(df_1w_low[1:] - df_1w_close[:-1])
    tr_1w = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index for 1w
    chop_raw = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        sum_atr = np.sum(atr_1w[i-13:i+1])  # sum of last 14 ATR values
        max_high = np.max(df_1w_high[i-13:i+1])
        min_low = np.min(df_1w_low[i-13:i+1])
        if max_high > min_low:
            chop_raw[i] = 100 * np.log10(sum_atr) / (np.log10(14) * np.log10(max_high - min_low))
        else:
            chop_raw[i] = 50.0  # neutral when no range
    
    # Calculate ATR(20) for 12h trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate volume MA(50) for 12h confirmation
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 12h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw, additional_delay_bars=0)  # CHOP is contemporaneous
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50, 20)  # Need enough bars for TRIX, CHOP, ATR/Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # TRIX signals: zero cross with slope
        trix_now = trix_aligned[i]
        trix_prev = trix_aligned[i-1] if i > 0 else 0.0
        trix_rising = trix_now > trix_prev
        trix_falling = trix_now < trix_prev
        
        if position == 0:
            # Volume confirmation
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            # Choppiness regime: only trade when CHOP < 38.2 (trending)
            chop_regime = chop_aligned[i] < 38.2
            
            # Long: TRIX crosses above zero AND rising AND volume confirmed AND trending regime
            if trix_now > 0 and trix_prev <= 0 and trix_rising and vol_confirmed and chop_regime:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: TRIX crosses below zero AND falling AND volume confirmed AND trending regime
            elif trix_now < 0 and trix_prev >= 0 and trix_falling and vol_confirmed and chop_regime:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.5*ATR
            if curr_close < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR
            if curr_close > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0