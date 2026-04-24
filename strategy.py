#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for lower trade frequency and reduced fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Donchian: 20-period upper/lower bands on 12h for breakout signals.
- Volume: Current 12h volume > 1.5 * 20-period 12h volume MA to confirm institutional interest.
- Entry: Long when price breaks above Donchian upper AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below Donchian lower AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Donchian band touch (long exits at lower band, short exits at upper band).
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy captures medium-term trends with clear structure, uses volume to filter false breakouts,
and avoids counter-trend trades via 1d EMA34 filter. Works in bull markets via long breakouts and
in bear markets via short breakouts, with volume confirmation reducing whipsaws.
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
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period 12h volume MA
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA
    volume_spike = volume > (1.5 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 20)  # Need enough bars for Donchian, EMA34, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        donchian_upper = highest_high[i]
        donchian_lower = lowest_low[i]
        ema_val = ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price above upper Donchian AND 1d EMA34 bullish (close > EMA)
                if curr_close > donchian_upper and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below lower Donchian AND 1d EMA34 bearish (close < EMA)
                elif curr_close < donchian_lower and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price touches or breaks below lower Donchian band
            if curr_low <= donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches or breaks above upper Donchian band
            if curr_high >= donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0