#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR(14) volatility filter.
- Long when price breaks above Donchian upper band (20-period high) AND 1w close > 1w EMA50 AND ATR(14) > 0.5 * ATR(50) (volatility expansion)
- Short when price breaks below Donchian lower band (20-period low) AND 1w close < 1w EMA50 AND ATR(14) > 0.5 * ATR(50)
- Exit on opposite Donchian band (lower for long exit, upper for short exit)
- Uses 1d primary with 1w HTF to target 30-100 trades over 4 years (7-25/year)
- Donchian provides clear breakout structure; EMA50 filters regime; ATR expansion confirms momentum
- Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1w_aligned
    bearish_regime = close < ema_50_1w_aligned
    
    # ATR(14) and ATR(50) for volatility expansion filter
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # first value
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility expansion: ATR(14) > 0.5 * ATR(50) (recent volatility above half of long-term)
    vol_expansion = atr_14 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14, 50)  # Need Donchian, EMA50, ATR14, ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper AND bullish regime AND volatility expansion
            if close[i] > donchian_upper[i] and bullish_regime[i] and vol_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND bearish regime AND volatility expansion
            elif close[i] < donchian_lower[i] and bearish_regime[i] and vol_expansion[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian lower (opposite band)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian upper (opposite band)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_ATRExpansion_v1"
timeframe = "1d"
leverage = 1.0