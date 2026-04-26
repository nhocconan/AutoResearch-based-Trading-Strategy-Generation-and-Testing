#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_VolumeSpike
Hypothesis: 4h strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI extremes and choppiness regime filter for mean reversion entries in ranging markets. Volume spike confirms momentum. Designed to work in both bull and bear markets by adapting to regime: in chop (CHOP>61.8) use RSI mean reversion at extremes; in trend (CHOP<38.2) follow KAMA direction. Target 20-50 trades/year to avoid fee drag.
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
    
    # Load daily data for ATR (used in choppiness calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # --- Indicators ---
    # KAMA for trend direction (using 1d close for stability)
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period sum of absolute changes
    # Pad volatility to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI(14) on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to align with close (first 14 values are NaN)
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index (CHOP) on 1d data
    atr_period = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    # Highest high and lowest low over atr_period
    hh = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.divide(
        atr * np.sqrt(atr_period),
        (hh - ll),
        out=np.full_like(atr, np.nan),
        where=(hh - ll)!=0
    )
    chop *= 100
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 14 for RSI, 14 for CHOP, 50 for KAMA)
    start_idx = max(20, 14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime determination
        is_chop = chop_val > 61.8   # Ranging market
        is_trend = chop_val < 38.2  # Trending market
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if is_chop:
            # In chop: mean reversion at RSI extremes
            long_entry = (rsi_val < 30) and vol_spike
            short_entry = (rsi_val > 70) and vol_spike
        elif is_trend:
            # In trend: follow KAMA direction
            long_entry = (close_val > kama_val) and vol_spike
            short_entry = (close_val < kama_val) and vol_spike
        # In transition zone (38.2 <= CHOP <= 61.8): no entries
        
        # Exit conditions: opposite signal or regime change to chop with RSI normalization
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: RSI > 50 in chop, or price < KAMA in trend, or volume spike reversal
            if is_chop and rsi_val > 50:
                exit_long = True
            elif is_trend and close_val < kama_val:
                exit_long = True
            # Also exit if RSI reaches opposite extreme in chop
            elif is_chop and rsi_val > 70:
                exit_long = True
        elif position == -1:
            # Exit short: RSI < 50 in chop, or price > KAMA in trend, or volume spike reversal
            if is_chop and rsi_val < 50:
                exit_short = True
            elif is_trend and close_val > kama_val:
                exit_short = True
            # Also exit if RSI reaches opposite extreme in chop
            elif is_chop and rsi_val < 30:
                exit_short = True
        
        # Minimum holding period: 1 bar to avoid whipsaw
        min_hold = 1
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0