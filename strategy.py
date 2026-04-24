#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend filter.
- Donchian breakout: long when price > highest high of last 20 bars, short when price < lowest low of last 20 bars.
- Trend filter: only trade long if 1d EMA34 is rising, short if falling.
- Volume confirmation: current volume > 2.0x 20-period volume MA to avoid low-volatility false breakouts.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Stoploss: exit position when price retraces to 10-period EMA on 4h.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # 4h indicators for Donchian breakout and exit
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA10 for exit signal
    ema_10_4h = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20, 10)  # EMA34 + Donchian + EMA10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_10_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout in direction of 1d EMA34 trend
            if i > 0 and not np.isnan(ema_34_1d_aligned[i-1]):
                ema34_slope = ema_34_1d_aligned[i] - ema_34_1d_aligned[i-1]
                if ema34_slope > 0:  # Uptrend - look for long breakout
                    if close[i] > donchian_high[i-1] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend - look for short breakdown
                    if close[i] < donchian_low[i-1] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long: hold until price retraces to 4h EMA10 or opposite breakdown
            if close[i] < ema_10_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: hold until price retraces to 4h EMA10 or opposite breakout
            if close[i] > ema_10_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0