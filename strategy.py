#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v2
Hypothesis: Daily KAMA trend direction combined with RSI extremes and choppiness regime filter.
KAMA adapts to market noise, reducing whipsaw in ranging conditions. RSI<30/>70 provides mean-reversion entries in choppy markets (CHOP>61.8), while KAMA trend filters for direction in trending markets (CHOP<38.2). Designed for low turnover (~15-25 trades/year) with edge in both bull/bear markets via regime-adaptive logic.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for KAMA trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = pd.Series(change).rolling(window=10, min_periods=10).sum() / \
         pd.Series(volatility).rolling(window=10, min_periods=10).sum()
    er = er.fillna(0).values
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1w data for choppiness regime (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) and Sum of ATR for CHOP
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Max/Min close over 14 periods for CHOP denominator
    max_close_1w = pd.Series(close_1w).rolling(window=14, min_periods=14).max().values
    min_close_1w = pd.Series(close_1w).rolling(window=14, min_periods=14).min().values
    range_1w = max_close_1w - min_close_1w
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR) / (max(H)-min(L))) / log10(14)
    chop_raw = 100 * np.log10(sum_atr_1w / range_1w) / np.log10(14)
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw, additional_delay_bars=0)
    
    # RSI(14) on 1d for mean-reversion signals
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d KAMA (30) and RSI (14)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Regime filters
        chop = chop_aligned[i]
        is_choppy = chop > 61.8  # Range market -> mean revert
        is_trending = chop < 38.2  # Trend market -> follow trend
        
        # Trend direction from KAMA
        uptrend = curr_close > kama_aligned[i]
        downtrend = curr_close < kama_aligned[i]
        
        if position == 0:
            # Look for entry signals
            if is_choppy:
                # In choppy market: mean reversion at RSI extremes
                if rsi_aligned[i] < 30 and uptrend:  # Oversold with bullish bias
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif rsi_aligned[i] > 70 and downtrend:  # Overbought with bearish bias
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            elif is_trending:
                # In trending market: follow KAMA direction
                if uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: exit conditions
            # Exit if RSI becomes overbought (take profit) or trend changes
            if rsi_aligned[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if RSI becomes oversold (take profit) or trend changes
            if rsi_aligned[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0