#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: 1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index regime filter.
Only takes trades when KAMA slope confirms trend, RSI is not extreme, and market is
trending (CHOP < 38.2) or mean-reverting (CHOP > 61.8) with appropriate RSI bias.
Designed for low trade frequency (7-25/year) to avoid fee drag while capturing
trends and mean reversals in both bull and bear markets.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on 1d close (requires 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # KAMA parameters: ER period=10, fast=2, slow=30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=er_period))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Handle first er_period values
    change_padded = np.concatenate([np.full(er_period, np.nan), change])
    volatility_padded = np.concatenate([np.full(er_period, np.nan), 
                                        pd.Series(np.abs(np.diff(close_1d))).rolling(window=er_period, min_periods=1).sum().values])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[er_period] = close_1d[er_period]  # seed
    for i in range(er_period + 1, len(close_1d)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = close_1d[i]
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    rsi_period = 14
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi_padded = np.concatenate([np.full(rsi_period, np.nan), rsi])
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_padded)
    
    # Calculate Choppiness Index(14) on 1d high/low/chop
    chop_period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # MaxHigh - MinLow over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    range_1d = max_high - min_low
    
    # Chop = 100 * log10(ATR_sum / range) / log10(chop_period)
    chop = np.where(range_1d > 0, 
                    100 * np.log10(atr_1d / range_1d) / np.log10(chop_period), 
                    50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA seed (er_period), RSI (rsi_period), Chop (chop_period), HTF EMA (50)
    start_idx = max(er_period, rsi_period, chop_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Determine regime based on Chop
            is_trending = chop_val < 38.2
            is_ranging = chop_val > 61.8
            
            # Long conditions
            long_cond = False
            if is_trending:
                # In trending regime: go with KAMA direction and HTF trend
                long_cond = (close_val > kama_val) and (close_val > ema_50_1w_val) and (rsi_val > 50) and (rsi_val < 70)
            elif is_ranging:
                # In ranging regime: mean reversion from extremes
                long_cond = (close_val < kama_val) and (rsi_val < 30)  # oversold
            
            # Short conditions
            short_cond = False
            if is_trending:
                # In trending regime: go with KAMA direction and HTF trend
                short_cond = (close_val < kama_val) and (close_val < ema_50_1w_val) and (rsi_val < 50) and (rsi_val > 30)
            elif is_ranging:
                # In ranging regime: mean reversion from extremes
                short_cond = (close_val > kama_val) and (rsi_val > 70)  # overbought
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            # Trend reversal: price crosses below KAMA or HTF trend turns bearish
            # RSI overbought exit in ranging market
            if ((close_val < kama_val) or 
                (close_val < ema_50_1w_val) or
                (chop_val > 61.8 and rsi_val > 70) or
                (chop_val < 38.2 and rsi_val > 80)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            # Trend reversal: price crosses above KAMA or HTF trend turns bullish
            # RSI oversold exit in ranging market
            if ((close_val > kama_val) or 
                (close_val > ema_50_1w_val) or
                (chop_val > 61.8 and rsi_val < 30) or
                (chop_val < 38.2 and rsi_val < 20)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0