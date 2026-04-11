#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: 1d KAMA trend direction with RSI momentum and 1-week Choppiness index regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, filtering out whipsaws. RSI confirms momentum strength.
# Choppiness index identifies ranging markets (CHOP > 61.8) for mean-reversion and trending markets (CHOP < 38.2) for trend-following.
# Works in bull markets by riding trends with momentum confirmation.
# Works in bear markets by mean-reverting in ranges and avoiding false trends.
# Uses 1-week CHOP for regime detection to reduce false signals and adapt to market conditions.
# Low frequency (~10-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week Choppiness Index calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(tr14) / (max_hh - min_ll)) / log10(14)
    range_14 = max_hh_14 - min_ll_14
    chop = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    chop = np.where(range_14 == 0, 100, chop)  # avoid division by zero
    chop = np.where(np.isnan(chop), 50, chop)  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(close - np.roll(close, 10))
    change = np.concatenate([[np.nan], change[1:]])  # align
    
    vol = np.abs(np.diff(close, prepend=np.nan))
    vol_sum = pd.Series(vol).rolling(window=10, min_periods=10).sum().values
    
    er = np.where(vol_sum != 0, change / vol_sum, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # when no losses
    rsi = np.where(avg_gain == 0, 0, rsi)    # when no gains
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: Choppiness index
        # CHOP > 61.8 = ranging market (mean revert)
        # CHOP < 38.2 = trending market (trend follow)
        ranging = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2
        
        # KAMA trend direction
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        rsi_bullish = 50 < rsi[i] < 70
        rsi_bearish = 30 < rsi[i] < 50
        
        # Entry logic
        if ranging:
            # In ranging markets: mean reversion at RSI extremes
            if rsi_oversold and price_above_kama and position != 1:
                position = 1
                signals[i] = 0.25
            elif rsi_overbought and price_below_kama and position != -1:
                position = -1
                signals[i] = -0.25
        elif trending:
            # In trending markets: follow trend with momentum
            if price_above_kama and rsi_bullish and position != 1:
                position = 1
                signals[i] = 0.25
            elif price_below_kama and rsi_bearish and position != -1:
                position = -1
                signals[i] = -0.25
        
        # Exit logic: opposite signal or regime change to extreme
        if position == 1 and (rsi_overbought or price_below_kama or chop_aligned[i] > 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_oversold or price_above_kama or chop_aligned[i] > 70):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals