#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and 1d volume spike.
- Camarilla H3/L3 levels act as intraday support/resistance; breaks indicate momentum.
- 4h EMA50 ensures trades align with intermediate trend, reducing whipsaws in chop.
- 1d volume spike (>2x 20-bar average) confirms conviction, filters low-quality breakouts.
- Position size 0.20 balances profit and drawdown; discrete levels minimize fee churn.
- Session filter (08-20 UTC) avoids low-liquidity Asian session noise.
- Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag on 1h timeframe.
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
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d volume MA for spike detection (>2x average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Camarilla levels from prior 1h bar (H3, L3, H4, L4)
    lookback = 1
    H1 = pd.Series(high).shift(lookback).values
    L1 = pd.Series(low).shift(lookback).values
    C1 = pd.Series(close).shift(lookback).values
    rng = H1 - L1
    
    # Camarilla H3/L3 (3/8 and 5/8 levels)
    H3 = C1 + rng * 1.1 / 4
    L3 = C1 - rng * 1.1 / 4
    H4 = C1 + rng * 1.1 / 2
    L4 = C1 - rng * 1.1 / 2
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 1) + 1  # Need EMA and prior bar data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session and volume confirmation filters
        session_ok = in_session[i]
        volume_spike = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Only trade if in session and volume confirms
            if session_ok and volume_spike:
                # Long breakout: price above H3 AND above 4h EMA50
                if close[i] > H3[i] and close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short breakout: price below L3 AND below 4h EMA50
                elif close[i] < L3[i] and close[i] < ema_50_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR crosses below 4h EMA50
            if close[i] < L3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR crosses above 4h EMA50
            if close[i] > H3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0