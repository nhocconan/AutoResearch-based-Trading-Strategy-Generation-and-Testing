#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing a smooth trend filter. In trending markets,
# price stays on one side of KAMA; in ranging markets, it crosses frequently. We use RSI for
# overbought/oversold conditions and Choppiness Index to filter ranging markets. Only trade
# in direction of KAMA trend when RSI is not extreme and market is trending (CHOP < 61.8).
# This avoids whipsaws in ranging markets and captures trends in both bull and bear markets.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_KAMA_Trend_With_RSI_and_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    atr_1d = []
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop = 100 * chop_denom
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)  # avoid division by zero
    chop = np.where(np.isnan(chop), 50, chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate KAMA (10-period ER, 2 and 30 SC)
    change = np.abs(np.concatenate([[close[0]], close[:-1]]) - close)
    direction = np.abs(close - np.concatenate([[close[0]], close[:-10]])) if len(close) > 10 else np.zeros_like(close)
    er = np.where(change != 0, direction / np.concatenate([[change[0]], change[:-10]]) if len(change) > 10 else 0, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)  # Use same length
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14), volume MA (20)
    start_idx = max(10, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend filter
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Choppiness filter: only trade in trending markets (CHOP < 61.8)
        trending_market = chop_aligned[i] < 61.8
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: above KAMA, RSI not overbought, trending market, volume
            if above_kama and rsi_not_overbought and trending_market and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: below KAMA, RSI not oversold, trending market, volume
            elif below_kama and rsi_not_oversold and trending_market and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: below KAMA or RSI overbought or ranging market
            if below_kama or not rsi_not_overbought or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: above KAMA or RSI oversold or ranging market
            if above_kama or not rsi_not_oversold or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals