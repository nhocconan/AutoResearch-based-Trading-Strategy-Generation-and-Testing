#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, KAMA trend direction + RSI mean reversion + Choppiness Index regime filter provides edge in both bull and bear markets.
- KAMA(10,2,30) defines adaptive trend: long when price > KAMA, short when price < KAMA
- RSI(14) < 40 for long entries, RSI(14) > 60 for short entries (mean reversion within trend)
- Choppiness Index(14) > 61.8 = ranging market (avoid trend following), < 38.2 = trending (allow entries)
- Volume confirmation: current volume > 1.5 * 20-day average volume
- Discrete position sizing: 0.25 to minimize fee drag
- Target: 15-25 trades/year per symbol (60-100 total over 4 years)
- Works in bull via trend filter, works in bear via mean reversion entries during pullbacks
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA(10,2,30) on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period sum of 1-period changes
    # Pad volatility to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start at period 10
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index(14) on 1d OHLC
    # True Range
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.concatenate([[np.nan], df_1d['close'].values[:-1]]))
    tr3 = np.abs(df_1d['low'].values - np.concatenate([[np.nan], df_1d['close'].values[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where((atr14 > 0) & (hh14 > ll14),
                    100 * np.log10(atr14 / (hh14 - ll14)) / np.log10(14),
                    50)  # Default to 50 when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade when trending (CHOP < 38.2) or extreme RSI in ranging (CHOP > 61.8)
        trending_regime = chop_aligned[i] < 38.2
        ranging_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long entry: price > KAMA (uptrend) AND RSI < 40 (oversold) AND volume spike
            # In trending regime: follow trend with mean reversion entry
            # In ranging regime: mean reversion at extremes
            if ((trending_regime and close[i] > kama_aligned[i]) or ranging_regime) and \
               rsi_aligned[i] < 40 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA (downtrend) AND RSI > 60 (overbought) AND volume spike
            elif ((trending_regime and close[i] < kama_aligned[i]) or ranging_regime) and \
                 rsi_aligned[i] > 60 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < KAMA (trend break) OR RSI > 70 (overbought) OR regime change to extreme ranging
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70 or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > KAMA (trend break) OR RSI < 30 (oversold) OR regime change to extreme ranging
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30 or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0