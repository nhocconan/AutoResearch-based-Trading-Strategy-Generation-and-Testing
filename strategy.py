#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter, volume spike confirmation, and session filter (08-20 UTC).
- Uses discrete position size 0.20 to limit drawdown and reduce fee churn.
- Volume confirmation requires >2.0x 24-period average to ensure conviction.
- 4h EMA34 trend filter ensures alignment with higher timeframe momentum.
- Session filter reduces noise trades during low-liquidity hours.
- Designed for 15-30 trades/year (60-120 total over 4 years) to stay within fee-efficient range.
- Combines Camarilla structure with 4h trend filter and volume confirmation for robustness.
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
    
    # Calculate Camarilla levels
    camarilla_h3 = close_4h_aligned + 1.1 * (high_4h_aligned - low_4h_aligned) / 4
    camarilla_l3 = close_4h_aligned - 1.1 * (high_4h_aligned - low_4h_aligned) / 4
    
    # 4h EMA34 trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average) + session filter
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        session_ok = in_session[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 4h EMA34 AND volume confirmation AND session
            if close[i] > camarilla_h3[i] and close[i] > ema_34_4h_aligned[i] and volume_confirm and session_ok:
                signals[i] = 0.20
                position = 1
            # Short: Close < L3 AND price below 4h EMA34 AND volume confirmation AND session
            elif close[i] < camarilla_l3[i] and close[i] < ema_34_4h_aligned[i] and volume_confirm and session_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close < L3 OR price crosses below 4h EMA34
            if close[i] < camarilla_l3[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close > H3 OR price crosses above 4h EMA34
            if close[i] > camarilla_h3[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0