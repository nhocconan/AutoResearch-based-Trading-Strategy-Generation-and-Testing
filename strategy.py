#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price breaks above Donchian(20) high AND 12h EMA trend bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 12h EMA trend bearish AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian breakouts work in both bull and bear markets by capturing strong momentum moves,
while EMA filter avoids counter-trend trades and volume confirmation reduces false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    df_12h_close = df_12h['close'].values
    ema_50_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Determine 12h trend: 1 if bullish (close > EMA50), -1 if bearish (close < EMA50), 0 otherwise
    trend_12h = np.where(df_12h_close > ema_50_12h, 1, np.where(df_12h_close < ema_50_12h, -1, 0))
    
    # Align HTF indicators to 4h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need enough bars for Donchian and 12h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(trend_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume[i] > (2.0 * vol_ma[i])
        trend = trend_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and trend alignment
            if vol_spike:
                # Bullish: price breaks above Donchian high AND 12h trend bullish
                if curr_high > period20_high[i] and trend == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low AND 12h trend bearish
                elif curr_low < period20_low[i] and trend == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation
            if curr_low < period20_low[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation
            if curr_high > period20_high[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0