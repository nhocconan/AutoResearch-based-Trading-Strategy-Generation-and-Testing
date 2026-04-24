#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based volatility filter.
- Primary timeframe: 1d for lower trade frequency and better signal quality in bear markets.
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volatility filter: Current 1d ATR(14) < 1.5 * 20-period ATR(14) MA to avoid high-volatility chop.
- Donchian: Upper = 20-period high, Lower = 20-period low from prior day.
- Entry: Long when price breaks above Upper AND 1w EMA50 bullish AND low volatility filter.
         Short when price breaks below Lower AND 1w EMA50 bearish AND low volatility filter.
- Exit: Price reverts to 20-period midpoint (mean reversion) or volatility filter fails.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
This strategy captures breakouts in the direction of the weekly trend while avoiding
false signals during high volatility periods. Works in both bull and bear markets by
only taking trades aligned with the 1w trend, with volatility filter ensuring
structure during ranging periods. Donchian breakouts provide clear entry/exit levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    tr1 = np.abs(df_1d_high - df_1d_low)
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First bar has no prior close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period 1d ATR MA for volatility filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian levels from prior 20 days
    # Upper = 20-period high, Lower = 20-period low
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    donchian_upper = pd.Series(h1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(l1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2  # Mean reversion exit
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volatility filter: current ATR < 1.5 * ATR MA (low volatility regime)
    low_volatility = atr_1d_aligned < (1.5 * atr_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(low_volatility[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with low volatility filter
            if low_volatility[i]:
                # Bullish breakout: price > Upper AND 1w EMA50 bullish (close > EMA)
                if curr_close > donchian_upper_aligned[i] and curr_close > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < Lower AND 1w EMA50 bearish (close < EMA)
                elif curr_close < donchian_lower_aligned[i] and curr_close < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint OR volatility filter fails
            if curr_close <= donchian_mid_aligned[i] or not low_volatility[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint OR volatility filter fails
            if curr_close >= donchian_mid_aligned[i] or not low_volatility[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolatilityFilter_v1"
timeframe = "1d"
leverage = 1.0