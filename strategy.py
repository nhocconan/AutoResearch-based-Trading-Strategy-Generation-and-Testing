#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA34 trend alignment (proven pattern from DB)
- Camarilla levels calculated from previous completed 1d bar's OHLC (based on prior daily candle)
- Long when price breaks above H3 AND price > 1d EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below L3 AND price < 1d EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the Camarilla H4/L4 midpoint (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 12h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Use previous completed 12h bar's OHLC for signal generation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Shift by 1 to use previous completed 12h bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar has no previous bar, set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate 1d EMA34 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous completed 1d bar's OHLC
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # We calculate on previous day data to avoid look-ahead
    lookback = 1
    h4 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    h3 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Use data up to previous bar (i-1) for Camarilla calculation
        start_idx = i - lookback
        end_idx = i  # exclusive, so we use [start_idx:end_idx] which is [i-1, i-1]
        if start_idx >= 0 and not (np.isnan(prev_high[start_idx:end_idx]).any() or np.isnan(prev_low[start_idx:end_idx]).any() or np.isnan(prev_close[start_idx:end_idx]).any()):
            ph = prev_high[start_idx:end_idx][0]
            pl = prev_low[start_idx:end_idx][0]
            pc = prev_close[start_idx:end_idx][0]
            rng = ph - pl
            h4[i] = pc + 1.5 * rng
            l4[i] = pc - 1.5 * rng
            h3[i] = pc + 1.125 * rng
            l3[i] = pc - 1.125 * rng
    
    # Midpoint for exit (between H4 and L4)
    midpoint = (h4 + l4) / 2.0
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(midpoint[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmation
            if close[i] > h3[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmation
            elif close[i] < l3[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint
            if close[i] < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint
            if close[i] > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0