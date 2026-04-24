#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend and Donchian channels.
- Donchian channels calculated from previous 20 1d bars.
- Entry: Long when price breaks above upper Donchian with volume spike and close > 1w EMA50 (uptrend).
         Short when price breaks below lower Donchian with volume spike and close < 1w EMA50 (downtrend).
- Exit: When price returns to the opposite Donchian level (mean reversion edge).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    lookback = 20
    upper_1d = np.full(n, np.nan)
    lower_1d = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper_1d[i] = np.max(high[i-lookback:i])
        lower_1d[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_1d[i]) or 
            np.isnan(lower_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > upper Donchian and close > 1w EMA50
                if close[i] > upper_1d[i] and close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower Donchian and close < 1w EMA50
                elif close[i] < lower_1d[i] and close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to lower Donchian (mean reversion)
            if close[i] <= lower_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to upper Donchian (mean reversion)
            if close[i] >= upper_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0