#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: 1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI extremes for entry timing and Choppiness Index for regime filtering.
Long when KAMA trending up, RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
Short when KAMA trending down, RSI > 70 (overbought), and CHOP > 61.8.
Exit on opposite RSI extreme (RSI > 70 for longs, RSI < 30 for shorts) or trend reversal.
Designed for low trade frequency (7-25/year) to minimize fee drag while capturing mean reversion
in ranging markets and trend continuation in strong moves. Works in both bull and bear markets
by adapting to regime via chop filter.
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
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d KAMA (trend direction) ===
    # KAMA parameters: ER period=10, Fast=2, Slow=30
    close_1d = pd.Series(df_1d['close'].values)
    change = abs(close_1d.diff(10).values)  # 10-period net change
    volatility = abs(close_1d.diff(1)).rolling(window=10, min_periods=10).sum().values  # 10-period sum of abs changes
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d.iloc[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 1d RSI (entry timing) ===
    # RSI(14) standard
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # === 1d Choppiness Index (regime filter) ===
    # CHOP(14) = 100 * log10(sum(ATR(1)) / (ATR(14) * sqrt(14))) / log10(sqrt(14))
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with 1d indexing
    atr1 = pd.Series(tr1).ewm(alpha=1/1, adjust=False).mean().values
    atr14 = pd.Series(tr1).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    chop_raw = 100 * np.log10(atr1.rolling(window=14, min_periods=14).sum() / (atr14 * np.sqrt(14))) / np.log10(np.sqrt(14))
    chop_values = chop_raw  # already aligned to 1d
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Look for entry: RSI extremes in ranging market
            # Long: RSI < 30 (oversold) AND KAMA trending up (price > KAMA)
            long_condition = in_range and (rsi_val < 30) and (close_val > kama_val)
            # Short: RSI > 70 (overbought) AND KAMA trending down (price < KAMA)
            short_condition = in_range and (rsi_val > 70) and (close_val < kama_val)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 (overbought) OR trend reversal (price < KAMA)
            exit_condition = (rsi_val > 70) or (close_val < kama_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 30 (oversold) OR trend reversal (price > KAMA)
            exit_condition = (rsi_val < 30) or (close_val > kama_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0