#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for structure and trend alignment.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 1.8 * 20-period volume MA to confirm institutional participation.
- Donchian: 20-period upper/lower bands for breakout signals.
- Entry: Long when price > upper band AND 12h EMA50 bullish AND volume spike.
         Short when price < lower band AND 12h EMA50 bearish AND volume spike.
- Exit: Opposite Donchian band touch (price < middle for long, price > middle for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
This captures medium-term momentum with trend filtering to avoid false breakouts,
while volume spikes confirm breakout validity. Works in bull markets via breakouts
and in bear markets via short breakdowns, with trend filter reducing whipsaws.
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
    
    # Calculate 4h Donchian(20) bands
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle = (upper + lower) / 2.0
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 12h volume MA
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)  # Need enough bars for Donchian, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > upper band AND 12h EMA50 bullish (close > EMA)
                if curr_close > upper[i] and curr_close > ema_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < lower band AND 12h EMA50 bearish (close < EMA)
                elif curr_close < lower[i] and curr_close < ema_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price < middle band OR loss of volume confirmation
            if curr_close < middle[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > middle band OR loss of volume confirmation
            if curr_close > middle[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0