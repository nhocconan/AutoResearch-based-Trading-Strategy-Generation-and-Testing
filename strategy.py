#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR-based volatility filter.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend and ATR calculation.
- Donchian levels from prior 12h: Upper = max(high,20), Lower = min(low,20) of prior 12h bar.
- Long when price breaks above Donchian Upper with ATR > 0.5 * ATR MA(20) (volatility filter),
  Short when price breaks below Donchian Lower with same volatility filter.
- Trend filter: Only trade in direction of 12h EMA50 (long if EMA50 rising, short if falling).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakouts in downtrend.
- Volatility filter ensures trades occur during sufficient momentum, avoiding choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for EMA50 trend filter, ATR, and Donchian levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50 and ATR
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(14) on 12h for volatility filter
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr2 = np.maximum(np.abs(low_12h[1:] - close_12h[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First TR is NaN
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_12h = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_12h > (0.5 * atr_ma_12h)  # ATR > 50% of its MA
    
    # Donchian(20) levels from prior 12h bar
    # Upper = max(high,20), Lower = min(low,20) of prior completed 12h bar
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels and volatility filter to 4h (each 12h bar = 3x 4h bars)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_12h, volatility_filter)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volatility_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 12h EMA50 trend
            if i > 0 and not np.isnan(ema_50_12h_aligned[i-1]):
                ema50_slope = ema_50_12h_aligned[i] - ema_50_12h_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Long when price breaks above Donchian Upper with volatility filter
                    if close[i] > donchian_upper_aligned[i] and volatility_filter_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Short when price breaks below Donchian Lower with volatility filter
                    if close[i] < donchian_lower_aligned[i] and volatility_filter_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian Lower or opposite signal
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian Upper or opposite signal
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0