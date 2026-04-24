#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation (>1.5x 24-bar average).
- Uses 4h for signal direction (trend filter) and 1h only for entry timing to reduce trade frequency.
- Session filter (08-20 UTC) to avoid low-liquidity hours.
- Discrete position size 0.20 to limit drawdown and fee churn.
- Targets 15-30 trades/year (60-120 total over 4 years) to stay fee-efficient.
- Works in bull/bear: 4h EMA50 ensures alignment with higher timeframe direction; volume filter avoids low-conviction entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Prior 4h OHLC (completed 4h bar)
    high_4h = df_4h['high'].shift(1).values
    low_4h = df_4h['low'].shift(1).values
    close_4h = df_4h['close'].shift(1).values
    
    # Align to 1h timeframe
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Calculate Camarilla levels (H3/L3) from prior 4h bar
    camarilla_h3 = close_4h_aligned + 1.1 * (high_4h_aligned - low_4h_aligned) / 4
    camarilla_l3 = close_4h_aligned - 1.1 * (high_4h_aligned - low_4h_aligned) / 4
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 4h EMA50 AND volume confirmation
            if close[i] > camarilla_h3[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: Close < L3 AND price below 4h EMA50 AND volume confirmation
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close < L3 OR price crosses below 4h EMA50
            if close[i] < camarilla_l3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close > H3 OR price crosses above 4h EMA50
            if close[i] > camarilla_h3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0