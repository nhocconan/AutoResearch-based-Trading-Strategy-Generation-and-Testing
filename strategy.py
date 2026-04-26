#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Filter_Chop_Regime
Hypothesis: On 4h timeframe, KAMA adapts to trend strength and choppy markets. When KAMA direction turns up/down with RSI confirmation (avoiding extremes) and chop regime filter (CHOP > 50 = range, < 50 = trend) we capture trend moves with minimal whipsaw. Works in both bull/bear markets: in trend (CHOP low) we follow KAMA+RSI; in range (CHOP high) we avoid false breakouts. Discrete sizing (±0.25) and ATR stoploss (2.5x) targets 25-40 trades/year.
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
    
    # --- Calculate KAMA (adaptive moving average) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    er[10:] = change[10:] / (volatility[10:] + 1e-10)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- Calculate RSI(14) ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi])  # align length
    
    # --- Calculate Choppiness Index (CHOP) ---
    # True Range over 14 periods
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop = np.concatenate([[np.nan] * 13, chop])  # align length
    
    # --- Load 1d data for regime filter (optional HTF trend) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # ATR for stoploss
    tr_atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr_atr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Warmup: max of KAMA seed(10), RSI(14), CHOP(13), ATR(20)
    start_idx = max(10, 14, 13, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_1d_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(ema_1d_val) or np.isnan(atr_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # KAMA direction: slope over 3 periods
        if i >= 3:
            kama_up = kama_val > kama[i-3]
            kama_down = kama_val < kama[i-3]
        else:
            kama_up = kama_down = False
        
        # RSI filter: avoid extremes, favor momentum
        rsi_bullish = 50 < rsi_val < 70  # not overbought, above midpoint
        rsi_bearish = 30 < rsi_val < 50  # not oversold, below midpoint
        
        # Chop regime: CHOP > 50 = range (avoid trend following), CHOP < 50 = trend (follow)
        # In range: we still follow KAMA+RSI but tighter? Actually, we use chop to avoid false signals in strong range
        # Better: only trade when CHOP < 60 (not extreme chop) to avoid whipsaw
        chop_filter = chop_val < 60  # allow some chop but not extreme
        
        # HTF trend filter: align with 1d EMA50
        htf_bullish = close_val > ema_1d_val
        htf_bearish = close_val < ema_1d_val
        
        # Entry conditions
        long_entry = kama_up and rsi_bullish and chop_filter and htf_bullish
        short_entry = kama_down and rsi_bearish and chop_filter and htf_bearish
        
        # Update highest/lowest for trailing stop
        if position == 1:
            highest_since_long = max(highest_since_long, high[i])
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low[i])
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit: ATR trailing stop (2.5x)
        long_exit = False
        short_exit = False
        if position == 1:
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high[i]
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low[i]
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_KAMA_Direction_RSI_Filter_Chop_Regime"
timeframe = "4h"
leverage = 1.0