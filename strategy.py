#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: Use KAMA (10-period ER) to determine trend direction on daily timeframe,
# filtered by RSI(14) > 50 for longs and < 50 for shorts, with Choppiness Index(14) > 61.8
# to avoid trending markets where mean reversion fails. Designed for low-frequency,
# high-conviction trades in ranging markets, with position sizing 0.25 to manage drawdown.
# Works in both bull and bear markets by avoiding strong trends via chop filter.

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = [close[0]]  # initialize
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate Choppiness Index on weekly data
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh14 = df_1w['high'].rolling(window=14, min_periods=14).max()
    ll14 = df_1w['low'].rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(atr14 / (hh14 - ll14)) / log10(14)
    chop = 100 * np.log10(atr14 / (hh14 - ll14)) / np.log10(14)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when range=0
    
    # Align weekly Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Align daily KAMA and RSI to daily (no alignment needed, but for consistency)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    
    # Volume confirmation: current volume > 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Warmup for volume MA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # RSI filter: >50 for long bias, <50 for short bias
        rsi_long = rsi_aligned[i] > 50
        rsi_short = rsi_aligned[i] < 50
        
        # Chop filter: only trade when market is ranging (Chop > 61.8)
        ranging = chop_aligned[i] > 61.8
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price above KAMA, RSI > 50, ranging market, volume confirmation
            if above_kama and rsi_long and ranging and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA, RSI < 50, ranging market, volume confirmation
            elif below_kama and rsi_short and ranging and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI < 50 or market trends (Chop < 38.2)
            if not above_kama or not rsi_long or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI > 50 or market trends (Chop < 38.2)
            if not below_kama or not rsi_short or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals