#!/usr/bin/env python3
"""
6h_Keltner_RSI_MeanReversion_Bollinger
Hypothesis: In range-bound markets (common in 2025-2026 BTC/ETH), price reverts to the mean from Keltner Channel extremes. Uses RSI(2) for oversold/overbought and Bollinger Band width to filter ranging regimes. Works in both bull and bear by only trading when volatility is low (range regime), avoiding trending whipsaws.
"""

name = "6h_Keltner_RSI_MeanReversion_Bollinger"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Bollinger Band Width for Regime Filter (1d) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Bollinger Bands on daily close
    close_1d = df_1d['close'].values
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d  # Normalized width
    
    # Align BB width to 6t
    bb_width_6h = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Regime: ranging when BB width is low (below 20th percentile)
    # Use rolling percentile of BB width (50-period lookback)
    bb_width_series = pd.Series(bb_width_6h)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    ranging_regime = bb_width_percentile < 20  # Low volatility = ranging
    
    # === Keltner Channel (20, ATR=10) on 6h close ===
    atr_mult = 1.0
    atr_period = 10
    kc_period = 20
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Keltner Channel
    sma_close = pd.Series(close).rolling(window=kc_period, min_periods=kc_period).mean().values
    kc_upper = sma_close + atr_mult * atr
    kc_lower = sma_close - atr_mult * atr
    
    # === RSI(2) for Mean Reversion Signals ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    
    # Start after warmup (covers ATR, SMA, RSI, BB width)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(ranging_regime[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging regime (low volatility)
        if not ranging_regime[i]:
            signals[i] = 0.0
            continue
        
        # Long: Price at Keltner lower band + RSI oversold
        if close[i] <= kc_lower[i] and rsi[i] < 20:
            signals[i] = position_size
        # Short: Price at Keltner upper band + RSI overbought
        elif close[i] >= kc_upper[i] and rsi[i] > 80:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals