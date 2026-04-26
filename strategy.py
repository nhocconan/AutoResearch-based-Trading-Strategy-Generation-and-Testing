#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period MA). 
Long when price breaks above R1 with uptrend and volume spike. 
Short when price breaks below S1 with downtrend and volume spike. 
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to work in both bull and bear markets by following the 12h trend, which adapts to regime changes.
Target: 19-50 trades/year (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    uptrend_12h = close > ema_50_12h_aligned
    downtrend_12h = close < ema_50_12h_aligned
    
    # Calculate Camarilla levels from previous day
    # Need to group by date to get daily OHLC
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Shift to get previous day's levels
    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    daily['prev_close'] = daily['close'].shift(1)
    
    # Calculate Camarilla levels
    daily['R1'] = daily['prev_close'] + 1.1 * (daily['prev_high'] - daily['prev_low']) / 12
    daily['S1'] = daily['prev_close'] - 1.1 * (daily['prev_high'] - daily['prev_low']) / 12
    
    # Merge back to intraday data
    df = df.merge(daily[['date', 'R1', 'S1']], on='date', how='left')
    R1 = df['R1'].values
    S1 = df['S1'].values
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 12h EMA + 20 for volume MA + 1 for daily merge)
    start_idx = 71
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 12h uptrend and volume spike
            if close[i] > R1[i] and uptrend_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 12h downtrend and volume spike
            elif close[i] < S1[i] and downtrend_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: 12h trend changes to downtrend OR price breaks below S1 (mean reversion)
            if not uptrend_12h[i] or close[i] < S1[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 12h trend changes to uptrend OR price breaks above R1 (mean reversion)
            if not downtrend_12h[i] or close[i] > R1[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0