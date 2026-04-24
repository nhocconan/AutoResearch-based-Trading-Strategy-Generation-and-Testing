#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w EMA50 for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Volume: Current 1d volume > 1.5 * 20-period volume MA to avoid low-volume breakouts.
- Entry: Long when price breaks above upper Donchian(20) AND 1w EMA50 bullish AND volume spike.
         Short when price breaks below lower Donchian(20) AND 1w EMA50 bearish AND volume spike.
- Exit: Opposite Donchian level (lower for long, upper for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Donchian channels provide robust breakout levels. Combined with weekly trend and volume filters,
this avoids false breakouts and works in both bull and bear markets by only taking trades
in the direction of the 1w trend.
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
    
    # Calculate Donchian(20) channels for 1d (based on previous 20 bars' high/low)
    # Upper = max(high of last 20 bars)
    # Lower = min(low of last 20 bars)
    # Using rolling window to avoid look-ahead
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 1w
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 1d volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1w_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above upper Donchian AND 1w EMA50 bullish (price > EMA)
                if curr_high > donchian_upper[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below lower Donchian AND 1w EMA50 bearish (price < EMA)
                elif curr_low < donchian_lower[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR loss of volume confirmation
            if curr_low < donchian_lower[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR loss of volume confirmation
            if curr_high > donchian_upper[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0