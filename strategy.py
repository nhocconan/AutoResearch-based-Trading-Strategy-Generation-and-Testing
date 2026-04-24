#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for structure and lower fee drag.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian: 20-period high/low breakout for momentum entries.
- Volume: Current 4h volume > 1.8 * 20-period volume MA to filter weak breakouts.
- Signal: Long on Donchian high breakout + 12h EMA50 bullish + volume spike.
          Short on Donchian low breakdown + 12h EMA50 bearish + volume spike.
- Exit: Opposite Donchian breakout (long exits on low break, short exits on high break) or loss of volume/trend.
- Signal size: 0.25 discrete to balance reward and risk while minimizing fee churn.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
This captures strong momentum moves in the direction of the 12h trend, filtered by volume to avoid false breakouts.
Works in bull markets (long bias) and bear markets (short bias) by following the 12h EMA50 trend.
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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
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
    start_idx = max(donchian_window, 50, 20)  # Need enough bars for Donchian, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Break above Donchian high AND 12h EMA50 bullish (close > EMA)
                if curr_high > donchian_high[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Break below Donchian low AND 12h EMA50 bearish (close < EMA)
                elif curr_low < donchian_low[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Break below Donchian low OR loss of volume confirmation OR loss of trend
            if curr_low < donchian_low[i] or not volume_spike[i] or curr_close < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Break above Donchian high OR loss of volume confirmation OR loss of trend
            if curr_high > donchian_high[i] or not volume_spike[i] or curr_close > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0