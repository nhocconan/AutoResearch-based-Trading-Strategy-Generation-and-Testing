#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA50 for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Volume: Current 4h volume > 1.3 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above Donchian upper band (20-period high) AND 1d EMA50 bullish AND volume spike.
         Short when price breaks below Donchian lower band (20-period low) AND 1d EMA50 bearish AND volume spike.
- Exit: Opposite Donchian band (lower for long, upper for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide objective trend-following structure. Combined with daily trend and volume filters,
this strategy captures strong momentum moves while avoiding false breakouts in choppy markets.
Works in both bull and bear markets by only taking trades in the direction of the daily trend.
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
    
    # Calculate Donchian channels (20-period) using previous bar's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Donchian upper band = 20-period high of previous highs
    # Donchian lower band = 20-period low of previous lows
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_upper[i] = np.max(prev_high[i-lookback:i])
        donchian_lower[i] = np.min(prev_low[i-lookback:i])
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.3 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need enough bars for EMA50, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above Donchian upper AND 1d EMA50 bullish (price > EMA)
                if curr_high > donchian_upper[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian lower AND 1d EMA50 bearish (price < EMA)
                elif curr_low < donchian_lower[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR loss of volume confirmation
            if curr_low < donchian_lower[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR loss of volume confirmation
            if curr_high > donchian_upper[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0