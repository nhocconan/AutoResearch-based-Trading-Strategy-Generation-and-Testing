#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for trend filter (price above/below EMA50).
- Entry: Long when price breaks above R1 with volume spike AND 4h EMA50 uptrend.
         Short when price breaks below S1 with volume spike AND 4h EMA50 downtrend.
- Exit: Opposite breakout (price crosses back below R1 for longs, above S1 for shorts).
- Signal size: 0.20 discrete to minimize fee drag.
- Camarilla levels provide intraday support/resistance; breakouts with volume indicate institutional participation.
- 4h EMA50 filter ensures trading with the higher timeframe trend.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema50_4h = ema(df_4h['close'].values, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels on 1h (based on previous day's range)
    # We'll use rolling window of 24 periods (1 day of 1h data) to simulate daily OHLC
    lookback = 24  # 24 * 1h = 1 day
    if n < lookback:
        return np.zeros(n)
    
    # Calculate rolling daily OHLC
    daily_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    daily_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    daily_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rng = daily_high - daily_low
    r1 = daily_close + rng * 1.1 / 12
    s1 = daily_close - rng * 1.1 / 12
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Ensure Camarilla and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_in_session = in_session[i]
        
        # Exit conditions: price crosses back below R1 for longs, above S1 for shorts
        if position != 0:
            # Exit long: price falls back below R1
            if position == 1:
                if curr_close < r1[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises back above S1
            elif position == -1:
                if curr_close > s1[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: breakout with volume spike and trend filter, only in session
        if position == 0 and curr_in_session:
            # Long: price breaks above R1 with volume spike AND 4h EMA50 uptrend
            if (curr_close > r1[i] and 
                volume_spike[i] and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike AND 4h EMA50 downtrend
            elif (curr_close < s1[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0