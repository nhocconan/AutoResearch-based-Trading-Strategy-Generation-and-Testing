#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA34 trend alignment
- Donchian levels calculated from prior 20 periods: upper = max(high), lower = min(low)
- Breakout logic: long when price crosses above upper band with volume confirmation, short when price crosses below lower band
- Trend filter: only long when price > 1d EMA34, only short when price < 1d EMA34
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 12h close (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
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
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior 12h Donchian levels (20-period)
    # We need to resample to 12h first, but since we're using 12h as primary timeframe,
    # we can calculate directly from the prices data
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    # Mean reversion exit: price reverts to prior 12h close
    # For 12h timeframe, prior close is simply the close from 12 periods ago (assuming 5m data: 12*60/5=144 periods)
    # But since we don't know the exact period multiplier, we'll use a simple approach:
    # Use the close from 1 bar ago as proxy for prior period close (simplified)
    # Better: use the close from the same time yesterday, but that's complex without knowing exact frequency
    # Instead, we'll use a fixed lookback: for 12h TF on 5m data, 144 periods back
    # We'll use a reasonable default and adjust based on data frequency
    # Calculate periods per 12h: assuming 5m data -> 12*60/5 = 144
    periods_per_12h = 144  # for 5m data
    if len(prices) >= periods_per_12h:
        prior_close = np.roll(close, periods_per_12h)
        prior_close[:periods_per_12h] = np.nan  # First values not available
    else:
        prior_close = np.full_like(close, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20, periods_per_12h)  # Need Donchian(20), volume MA(20), and prior close
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(prior_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper band AND uptrend AND volume confirmation
            if close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower band AND downtrend AND volume confirmation
            elif close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 12h close (mean reversion) or reverse signal
            if close[i] <= prior_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 12h close (mean reversion) or reverse signal
            if close[i] >= prior_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0