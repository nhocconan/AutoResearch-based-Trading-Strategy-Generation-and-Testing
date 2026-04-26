#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter_v3
Hypothesis: Trade daily KAMA trend with RSI extremes filtered by choppiness index.
KAMA adapts to market noise, reducing whipsaw in choppy regimes. RSI>70 or <30
provides mean-reversion entries in trending markets. Choppiness index (CHOP>61.8)
activates mean-reversion mode; CHOP<38.2 activates trend-following mode.
Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
Uses proven winning formula: adaptive trend + momentum filter + regime filter.
Works in bull markets via trend-following entries and in bear markets via
mean-reversion swings at RSI extremes during high-chop regimes.
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
    
    # Get 1w data for EMA34 trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate KAMA(10,2,30) - adaptive trend
    # ER = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation:
    diff = np.diff(close, prepend=close[0])
    abs_diff = np.abs(diff)
    change_over_period = np.abs(close - np.roll(close, 10))
    change_over_period[0:10] = np.nan
    sum_abs_diff = pd.Series(abs_diff).rolling(window=10, min_periods=10).sum().values
    er = np.where(sum_abs_diff > 0, change_over_period / sum_abs_diff, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed at period 10
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(TR) / (ATR * N)) / log10(N)
    atr_14 = atr  # already calculated
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = 0
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_tr / (atr_14 * 14)) / np.log10(14))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA seed, RSI, CHOP, ATR, EMA warmup
    start_idx = max(10, 14, 14, 14, 34) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(atr[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        atr_val = atr[i]
        trend_1w_up = close_val > ema_34_1w_aligned[i]
        trend_1w_down = close_val < ema_34_1w_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if chop_val > 61.8:  # High chop = mean reversion regime
                # Long: RSI < 30 (oversold) AND price > KAMA (bullish bias)
                long_signal = (rsi_val < 30) and (close_val > kama_val)
                # Short: RSI > 70 (overbought) AND price < KAMA (bearish bias)
                short_signal = (rsi_val > 70) and (close_val < kama_val)
            elif chop_val < 38.2:  # Low chop = trending regime
                # Long: price > KAMA AND 1w trend up
                long_signal = (close_val > kama_val) and trend_1w_up
                # Short: price < KAMA AND 1w trend down
                short_signal = (close_val < kama_val) and trend_1w_down
            else:  # Neutral chop = no new entries
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions: 
            # 1. Regime shift to high chop AND RSI > 50 (exit mean reversion)
            # 2. Trend flip (1w trend down)
            # 3. Stoploss hit
            if ((chop_val > 61.8 and rsi_val > 50) or
                (not trend_1w_up) or
                (close_val < entry_price - 2.5 * atr_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions:
            # 1. Regime shift to high chop AND RSI < 50 (exit mean reversion)
            # 2. Trend flip (1w trend up)
            # 3. Stoploss hit
            if ((chop_val > 61.8 and rsi_val < 50) or
                (not trend_1w_down) or
                (close_val > entry_price + 2.5 * atr_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Filter_v3"
timeframe = "1d"
leverage = 1.0