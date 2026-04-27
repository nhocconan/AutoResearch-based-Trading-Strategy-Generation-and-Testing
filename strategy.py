#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: Daily timeframe strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for momentum confirmation and Choppiness Index for regime filtering.
Only takes trades when KAMA slope indicates trend, RSI confirms momentum (not extreme),
and market is not too choppy (CHOP > 38.2). Designed to work in both bull and bear markets
by adapting to trend conditions while avoiding whipsaws in ranging markets. Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe trend filter (optional regime)
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0  # 1-period volatility
    # Fix: vol needs to be rolling sum of absolute changes
    vol_series = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=1).sum().values
    vol_series[:9] = np.nan  # first 9 values invalid
    er = np.where(vol_series != 0, change / vol_series, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to daily (already daily, but ensure alignment)
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # identity align for same TF
    
    # RSI(14)
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan  # first 14 values invalid
    
    # Choppiness Index (14)
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (atr14 > 0) & (highest_high != lowest_low),
        100 * np.log10(atr14 * 14 / (highest_high - lowest_low)) / np.log10(14),
        50  # default when range is zero
    )
    chop[:13] = np.nan  # first 13 values invalid
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14), volume avg (20)
    start_idx = max(10, 14, 14, 20)  # 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(chop[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        vol_conf = volume_confirm[i]
        
        # KAMA slope: rising if current > previous, falling if current < previous
        kama_rising = kama_val > kama_aligned[i-1] if i > 0 else False
        kama_falling = kama_val < kama_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Enter long: KAMA rising, RSI not overbought (>50 and <70), not choppy (CHOP > 38.2), volume confirmation
            long_condition = (
                kama_rising and
                50 < rsi_val < 70 and
                chop_val > 38.2 and
                vol_conf
            )
            # Enter short: KAMA falling, RSI not oversold (<50 and >30), not choppy (CHOP > 38.2), volume confirmation
            short_condition = (
                kama_falling and
                30 < rsi_val < 50 and
                chop_val > 38.2 and
                vol_conf
            )
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: KAMA falling OR RSI overbought (>70) OR choppy market (CHOP < 38.2)
            if kama_falling or rsi_val >= 70 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA rising OR RSI oversold (<30) OR choppy market (CHOP < 38.2)
            if kama_rising or rsi_val <= 30 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0