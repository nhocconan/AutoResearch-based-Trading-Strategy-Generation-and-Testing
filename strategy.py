#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Uses 1h timeframe (primary) and 4h HTF for EMA50 trend alignment
- Camarilla levels calculated from prior 1d OHLC: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
- Breakout logic: long when price closes above H3 with volume spike and uptrend,
                  short when price closes below L3 with volume spike and downtrend
- Trend filter: only long when 1h close > 4h EMA50, only short when 1h close < 4h EMA50
- Volume confirmation: current 1h volume > 1.5 * 20-period 1h volume MA
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity hours
- Discrete signal size: 0.20 to minimize fee churn and manage drawdown
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) using DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate prior 1d Camarilla levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's Camarilla: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align to 1h timeframe (wait for 1d bar to close)
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Trend filter: 1h close vs 4h EMA50
    uptrend = close > ema_50_4h_aligned
    downtrend = close < ema_50_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need 4h EMA50 and sufficient volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3_1d_aligned[i]) or 
            np.isnan(camarilla_l3_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above H3 AND uptrend AND volume spike
            if close[i] > camarilla_h3_1d_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price closes below L3 AND downtrend AND volume spike
            elif close[i] < camarilla_l3_1d_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 1d L4 (mean reversion) or reverse signal
            camarilla_l4_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
            camarilla_l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
            if not np.isnan(camarilla_l4_1d_aligned[i]) and close[i] <= camarilla_l4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to prior 1d H4 (mean reversion) or reverse signal
            camarilla_h4_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
            camarilla_h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
            if not np.isnan(camarilla_h4_1d_aligned[i]) and close[i] >= camarilla_h4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0