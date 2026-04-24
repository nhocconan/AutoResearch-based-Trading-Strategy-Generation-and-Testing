#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA20 trend filter and volume spike confirmation.
- Uses 1h timeframe (primary) and 4h HTF for EMA20 trend alignment
- Camarilla levels calculated from prior 4h OHLC: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
- Breakout logic: long when price closes above H3 with volume spike and uptrend,
                  short when price closes below L3 with volume spike and downtrend
- Trend filter: only long when 1h close > 4h EMA20, only short when 1h close < 4h EMA20
- Volume confirmation: current 1h volume > 2.0 * 20-period 1h volume MA
- Discrete signal size: 0.20 to balance reward and risk, minimizing fee churn
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity hours
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
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
    
    # Precompute session filter (08:00-20:00 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h EMA20 for trend confirmation (faster than EMA34)
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate prior 4h Camarilla levels (H3, L3)
    # Need to shift 4h data by 1 to avoid look-ahead (use prior completed 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Prior 4h bar's Camarilla: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    camarilla_h3_4h = close_4h + 1.1 * (high_4h - low_4h) / 4
    camarilla_l3_4h = close_4h - 1.1 * (high_4h - low_4h) / 4
    
    # Align to 1h timeframe (wait for 4h bar to close)
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 1h close > 4h EMA20 for uptrend, < for downtrend
    uptrend = close > ema_20_4h_aligned
    downtrend = close < ema_20_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 4h EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_h3_4h_aligned[i]) or 
            np.isnan(camarilla_l3_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above H3 AND uptrend AND volume spike
            if close[i] > camarilla_h3_4h_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price closes below L3 AND downtrend AND volume spike
            elif close[i] < camarilla_l3_4h_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 4h L3 (mean reversion) or reverse signal
            if close[i] <= camarilla_l3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to prior 4h H3 (mean reversion) or reverse signal
            if close[i] >= camarilla_h3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA20_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0