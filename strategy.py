#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA50 trend alignment (proven pattern from DB)
- Camarilla levels calculated from previous completed 12h bar's OHLC (based on prior 12h candle)
- Long when price breaks above Camarilla H3 AND price > 1d EMA50 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below Camarilla L3 AND price < 1d EMA50 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the Camarilla H3/L3 midpoint (mean reversion structure)
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
    
    # Use previous completed 12h bar's OHLC for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Shift by 1 to use previous completed 12h bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(prices['open'].values, 1)
    # First bar has no previous bar, set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Calculate 1d EMA50 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous completed 12h bar's OHLC
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # Midpoint for exit: (H3 + L3)/2 = C (the close)
    range_val = prev_high - prev_low
    camarilla_H3 = prev_close + range_val * 1.1 / 4
    camarilla_L3 = prev_close - range_val * 1.1 / 4
    camarilla_mid = (camarilla_H3 + camarilla_L3) / 2.0  # equals prev_close
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50, volume MA(20), and previous bar data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i]) or 
            np.isnan(camarilla_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_H3[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_L3[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Camarilla midpoint (previous close)
            if close[i] < camarilla_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla midpoint (previous close)
            if close[i] > camarilla_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0