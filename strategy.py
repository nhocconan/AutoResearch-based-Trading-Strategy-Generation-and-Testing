#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 12h EMA50 for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Volume: Current 4h volume > 1.3 * 20-period volume MA to avoid low-volume breakouts.
- Entry: Long when price breaks above upper Donchian(20) AND 12h EMA50 bullish AND volume spike.
         Short when price breaks below lower Donchian(20) AND 12h EMA50 bearish AND volume spike.
- Exit: Opposite Donchian level (lower for long, upper for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide objective trend-following breakouts. Combined with trend and volume filters,
this avoids false breakouts and works in both bull and bear markets by only taking trades in the
direction of the 12h trend. The strategy has shown strong performance on SOLUSDT in prior tests.
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
    
    # Calculate Donchian(20) channels (using previous 20 bars to avoid look-ahead)
    # Upper = max(high of last 20 bars), Lower = min(low of last 20 bars)
    # We use rolling window on previous bar's data
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    
    # Calculate rolling max/min on shifted arrays
    upper_channel = pd.Series(high_shift).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 12h
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 1.3 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (1.3 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
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
                # Bullish: price breaks above upper channel AND 12h EMA50 bullish (price > EMA)
                if curr_high > upper_channel[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below lower channel AND 12h EMA50 bearish (price < EMA)
                elif curr_low < lower_channel[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower channel OR loss of volume confirmation
            if curr_low < lower_channel[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper channel OR loss of volume confirmation
            if curr_high > upper_channel[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0