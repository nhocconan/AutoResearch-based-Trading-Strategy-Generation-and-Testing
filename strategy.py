#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter_V2
Hypothesis: Daily KAMA trend with RSI mean reversion and choppiness filter.
KAMA adapts to market noise, reducing whipsaws in ranging markets (bearish 2025).
RSI extremes provide mean reversion entries within the trend.
Choppiness index (CHOP) ensures we only trade in trending regimes (CHOP < 38.2) or mean revert in ranging (CHOP > 61.8).
Weekly trend filter ensures alignment with higher timeframe momentum.
Target: 15-25 trades/year, discrete sizing 0.25, Sharpe > 0.5 on BTC/ETH.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA (Adaptive Moving Average) - reduces whipsaws in chop
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(change.shape) > 0 else np.sum(np.abs(np.diff(close, n=1)))
    # Fix: calculate properly for 1D arrays
    volatility = np.zeros_like(change)
    for i in range(len(change)):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0,i-9):i+1], n=1)))
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already daily, but for consistency)
    df_1d = get_htf_data(prices, '1d')
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14) for mean reversion entries
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) - identifies trending vs ranging markets
    # TR = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid cases
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for weekly EMA, 14 for RSI/CHOP)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Regime-based logic:
        # Trending regime (CHOP < 38.2): follow KAMA trend
        # Ranging regime (CHOP > 61.8): mean revert at RSI extremes
        # Transition zone (38.2 <= CHOP <= 61.8): stay flat
        
        if chop_val < 38.2:  # Trending regime
            # Follow weekly trend with KAMA as dynamic support/resistance
            if close_val > kama_val and close_val > ema_50_val:
                # Long signal in uptrend
                if position <= 0:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close_val < kama_val and close_val < ema_50_val:
                # Short signal in downtrend
                if position >= 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Mixed signals - reduce position or stay flat
                signals[i] = 0.0
                position = 0
        elif chop_val > 61.8:  # Ranging regime
            # Mean reversion at RSI extremes
            if rsi_val < 30:  # Oversold - long
                if position <= 0:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif rsi_val > 70:  # Overbought - short
                if position >= 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # RSI in middle - exit to flat
                signals[i] = 0.0
                position = 0
        else:
            # Transition zone - stay flat to avoid whipsaws
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Filter_V2"
timeframe = "1d"
leverage = 1.0