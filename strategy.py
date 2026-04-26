#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering.
Only take long when KAMA up, RSI > 50, and market is trending (CHOP < 38.2).
Only take short when KAMA down, RSI < 50, and market is trending (CHOP < 38.2).
Stay flat in ranging markets (CHOP >= 38.2) to avoid whipsaws.
Uses discrete position sizing (0.25) to minimize fee churn and ensure capital preservation.
Designed to work in both bull and bear markets via adaptive trend detection and regime filtering.
Target: 30-100 trades over 4 years (7-25/year) by requiring confluence of trend, momentum, and regime.
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
    
    # Load 1w data ONCE before loop for HTF trend filter (optional confirmation)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # Using close prices directly as we're on 1d timeframe
    close_s = pd.Series(close)
    # Efficiency ratio: |net change| / sum of absolute changes over 10 periods
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)  # Avoid division by zero
    # Smoothing constants: fastest = 2/(2+1) = 0.6667, slowest = 2/(30+1) = 0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = close_s.copy()
    for i in range(1, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    # Align KAMA to 1d timeframe (no additional delay needed for EMA-like indicator)
    kama_aligned = align_htf_to_ltf(close_s.index, close_s.index, kama_values)  # Self-alignment for same timeframe
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(close_s.index, close_s.index, rsi_values)
    
    # Calculate Choppiness Index(14)
    # True Range: max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero when hh == ll
    denominator = hh - ll
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(close_s.index, close_s.index, chop)
    
    # Regime filter: trending when CHOP < 38.2, ranging when CHOP >= 38.2
    trending_regime = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 14 for RSI/CHOP, 10 for KAMA efficiency ratio)
    start_idx = max(14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade in trending regime
        if trending_regime[i]:
            # Long conditions: KAMA up (price > KAMA), RSI > 50 (bullish momentum)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Short conditions: KAMA down (price < KAMA), RSI < 50 (bearish momentum)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position in trending regime
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
        else:
            # In ranging regime: stay flat to avoid whipsaws
            signals[i] = 0.0
            position = 0  # Force flat in ranging markets
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0