#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index (CHOP) for regime filtering.
Enter long when KAMA trending up, RSI > 50, and CHOP < 61.8 (trending regime).
Enter short when KAMA trending down, RSI < 50, and CHOP < 61.8.
Exit when trend reverses or CHOP > 61.8 (range regime) to avoid whipsaws.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 trades over 4 years (7-25/year).
Uses 1-week HTF for higher-timeframe trend alignment to improve robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA calculation (primary trend) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start at index 9 (need 10 bars for ER)
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI(14) calculation ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Choppiness Index (CHOP) calculation ---
    # True Range over 14 periods
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = np.sum(tr, axis=1)  # Sum over 14 periods (need to align)
    # ATR-like sum of TR
    tr_sum = np.full(n, np.nan)
    for i in range(14, n):
        tr_sum[i] = np.sum(tr[i-13:i+1])  # TR from i-13 to i
    # Highest high and lowest low over 14 periods
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(13, n):
        max_high[i] = np.max(high[i-12:i+1])
        min_low[i] = np.min(low[i-12:i+1])
    # CHOP formula
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if tr_sum[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # --- Load 1-week HTF data for trend alignment ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    # Weekly EMA(10) for HTF trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # --- Volume confirmation (20-period SMA) ---
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA stability, 14 for RSI/CHOP, 20 for volume)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_10_1w_aligned[i]) or np.isnan(avg_volume[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_1w_val = ema_10_1w_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # KAMA trend direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        # Regime filter: CHOP < 61.8 indicates trending regime (avoid range)
        trending_regime = chop_val < 61.8
        
        # Long logic: KAMA rising, RSI > 50, trending regime, volume confirmation, and price above weekly EMA
        long_condition = (kama_rising and 
                         rsi_val > 50 and 
                         trending_regime and 
                         volume_confirmed and 
                         close_val > ema_1w_val)
        # Short logic: KAMA falling, RSI < 50, trending regime, volume confirmation, and price below weekly EMA
        short_condition = (kama_falling and 
                          rsi_val < 50 and 
                          trending_regime and 
                          volume_confirmed and 
                          close_val < ema_1w_val)
        
        # Exit logic: trend reversal or range regime
        exit_long = (not kama_rising) or (rsi_val < 50) or (not trending_regime)
        exit_short = (not kama_falling) or (rsi_val > 50) or (not trending_regime)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0