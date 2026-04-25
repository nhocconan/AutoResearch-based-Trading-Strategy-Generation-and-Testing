#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime
Hypothesis: Kaufman Adaptive Moving Average (KAMA) filters noise and adapts to market volatility.
Combined with RSI extremes and Choppiness Index regime filter, this strategy captures medium-term
trends while avoiding false signals in ranging markets. Works in both bull (trend following) and
bear (mean reversion in chop) regimes by adapting position sizing based on market state.
Uses 1d timeframe with 1w EMA50 for higher timeframe trend filter.
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
    
    # Get 1d data for KAMA, RSI, and Chop calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    close_1d = df_1d['close'].values
    direction = np.abs(close_1d[-1] - close_1d[0]) if len(close_1d) > 1 else 0
    volatility = np.sum(np.abs(np.diff(close_1d))) if len(close_1d) > 1 else 1
    er = direction / (volatility + 1e-10)  # Efficiency Ratio
    er = np.clip(er, 0, 1)
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2  # Smoothing Constant
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi])  # pad for min_periods
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index(14) on 1d data
    atr_1d = []
    tr_1d = []
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
        tr_1d.append(tr)
    tr_1d = np.array(tr_1d)
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d = np.concatenate([[np.nan], atr_1d])  # align with close
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        if sum_atr_14[i] > 0 and (max_high_14[i] - min_low_14[i]) > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    chop = np.concatenate([np.full(14, np.nan), chop[14:]])  # align
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Regime filter: Choppiness Index
        # CHOP > 61.8 = ranging market (mean revert)
        # CHOP < 38.2 = trending market (trend follow)
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # Exit conditions: regime change or opposite signal
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI > 70 (overbought) OR regime shifts to ranging
                if rsi_val > 70 or is_ranging:
                    exit_signal = True
                # Also exit if price falls below KAMA in trending market
                elif is_trending and curr_close < kama_val:
                    exit_signal = True
                    
            elif position == -1:
                # Exit short: RSI < 30 (oversold) OR regime shifts to ranging
                if rsi_val < 30 or is_ranging:
                    exit_signal = True
                # Also exit if price rises above KAMA in trending market
                elif is_trending and curr_close > kama_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        if position == 0:
            # Determine market regime and trade accordingly
            if is_trending:
                # Trending market: follow KAMA direction
                long_condition = curr_close > kama_val and rsi_val > 50
                short_condition = curr_close < kama_val and rsi_val < 50
            elif is_ranging:
                # Ranging market: mean reversion at RSI extremes
                long_condition = rsi_val < 30 and curr_close > kama_val
                short_condition = rsi_val > 70 and curr_close < kama_val
            else:
                # Transition regime: wait for clarity
                long_condition = False
                short_condition = False
            
            # Higher timeframe trend filter: only trade in direction of 1w EMA50
            if long_condition and curr_close > ema_trend:
                signals[i] = 0.25
                position = 1
            elif short_condition and curr_close < ema_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0