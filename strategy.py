#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d for lower trade frequency (target 30-100 total trades over 4 years).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1d volume > 2.0 * 20-period volume MA to capture institutional interest.
- Donchian: Upper/lower bands calculated from 20-period high/low.
- Entry: Long when price breaks above upper band AND 1w EMA50 bullish AND volume spike.
         Short when price breaks below lower band AND 1w EMA50 bearish AND volume spike.
- Exit: Price reverts to 20-period midpoint or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures medium-term breakouts filtered by weekly trend and volume confirmation,
designed to work in both bull and bear markets by only taking trades in the direction of the
1w trend, with volume spikes confirming participation. Donchian breakouts provide clear entry
signals with natural mean-reversion exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1w volume MA
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period Donchian bands from 1d data
    # Upper band = 20-period high
    # Lower band = 20-period low
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > upper band AND 1w EMA50 bullish (close > EMA)
                if curr_close > donchian_upper[i] and curr_close > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < lower band AND 1w EMA50 bearish (close < EMA)
                elif curr_close < donchian_lower[i] and curr_close < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to middle band OR loss of volume confirmation
            if curr_close <= donchian_middle[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle band OR loss of volume confirmation
            if curr_close >= donchian_middle[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0