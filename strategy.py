#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout with 4h EMA34 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as key intraday pivot points on 1h timeframe.
Breakouts above H3 with volume confirmation and 4h EMA34 uptrend signal momentum longs.
Breakdowns below L3 with volume confirmation and 4h EMA34 downtrend signal momentum shorts.
Using 4h EMA34 as HTF trend filter ensures alignment with medium-term trend.
Volume spike confirms participation. Discrete sizing (0.0, ±0.20) minimizes fee churn.
Session filter (08-20 UTC) reduces noise trades. Target: 15-37 trades/year on 1h.
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
    open_time = prices['open_time'].values
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous day's OHLC
    # 1h timeframe: 24 bars per day
    lookback = 24  # number of 1h bars in 1 day
    
    # Calculate rolling max/min/close for prior day (excluding current bar)
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla levels using prior day's OHLC
    diff = prev_high - prev_low
    H3 = prev_close + diff * 1.1 / 4
    L3 = prev_close - diff * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5 * 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 4h EMA34 (uptrend)
            long_entry = (curr_close > H3[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < 4h EMA34 (downtrend)
            short_entry = (curr_close < L3[i]) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below L3 (mean reversion) OR trend change (price < EMA)
            if (curr_close < L3[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (mean reversion) OR trend change (price > EMA)
            if (curr_close > H3[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0