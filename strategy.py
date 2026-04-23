#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d ATR-based volatility breakout with 1w EMA34 trend filter and volume confirmation.
- Volatility breakout: price moves > 1.5x 1d ATR from prior 1d close (captures expansion moves)
- Trend filter: price > 1w EMA34 for longs, < 1w EMA34 for shorts
- Volume confirmation: > 1.5x 24-period average volume
- Session filter: 08-20 UTC to avoid low liquidity
- Discrete position size: 0.25 to minimize fee churn
- Target: 12-37 trades/year (50-150 over 4 years)
- Works in bull/bear via volatility expansion + trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.5x 24-period average (strict for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1d ATR(14) for volatility breakout
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: use high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Prior 1d close for breakout calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First bar: use its own close
    
    # Align 1d indicators to 12h timeframe (use prior completed 1d bar)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24, 14)  # EMA34, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14_aligned[i]) or
            np.isnan(prev_close_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility breakout: price move > 1.5x ATR from prior close
        breakout_up = close[i] > prev_close_1d_aligned[i] + 1.5 * atr_14_aligned[i]
        breakout_down = close[i] < prev_close_1d_aligned[i] - 1.5 * atr_14_aligned[i]
        
        if position == 0:
            # Long: volatility breakout up AND price > 1w EMA34 AND volume confirmation AND in session
            if breakout_up and volume_confirm and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: volatility breakout down AND price < 1w EMA34 AND volume confirmation AND in session
            elif breakout_down and volume_confirm and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: volatility breakout down OR price < 1w EMA34 (trend flip)
            if breakout_down or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: volatility breakout up OR price > 1w EMA34 (trend flip)
            if breakout_up or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolatilityBreakout_ATR14_1wEMA34_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0