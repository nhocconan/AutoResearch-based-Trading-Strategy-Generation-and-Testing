#!/usr/bin/env python3
"""
#101006 - 4h_RSI2_MeanReversion_WithRegimeFilter
Hypothesis: Mean reversion using RSI(2) extremes combined with 1d trend filter and volatility regime filter.
Works in both bull and bear markets by only taking mean-reversion trades when the higher timeframe trend
is aligned, avoiding counter-trend trades during strong moves. Uses Bollinger Band width percentile to
detect low-volatility regimes for higher probability mean reversion.
Target: 20-30 trades/year to minimize fee drag. Uses discrete position sizing (0.25).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA for trend filter on daily
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(2) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Band Width for volatility regime (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Percentile of BB width over 50 periods to identify low volatility regimes
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == 50 else np.nan, raw=False
    ).values
    
    # Low volatility regime: BB width below 30th percentile (tightening bands)
    low_vol_regime = bb_width_percentile < 30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(low_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: RSI(2) < 10 (oversold), price above daily EMA50 (uptrend filter), low volatility regime
        if (rsi[i] < 10 and 
            close[i] > ema50_1d_aligned[i] and 
            low_vol_regime[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: RSI(2) > 90 (overbought), price below daily EMA50 (downtrend filter), low volatility regime
        elif (rsi[i] > 90 and 
              close[i] < ema50_1d_aligned[i] and 
              low_vol_regime[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone (50) or volatility regime changes
        elif position == 1 and (rsi[i] >= 50 or not low_vol_regime[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi[i] <= 50 or not low_vol_regime[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI2_MeanReversion_WithRegimeFilter"
timeframe = "4h"
leverage = 1.0