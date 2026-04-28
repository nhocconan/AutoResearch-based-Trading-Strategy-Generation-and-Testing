#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-bar high) AND 1w EMA50 is rising AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower (20-bar low) AND 1w EMA50 is falling AND volume > 1.5x 20-bar avg
# Exits when price reverts to Donchian middle (10-bar average) or trend reverses
# Target: 7-25 trades/year via tight breakout conditions + trend filter to avoid whipsaw
# Works in bull markets via long breakouts, in bear markets via short breakouts
# Uses 1w HTF for major trend alignment to reduce false signals during corrections/rallies

name = "1d_Donchian20_1wEMA50_Trend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (properly delayed for completed weekly bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 1d data
    # Upper = 20-period high, Lower = 20-period low, Middle = 10-period average
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = close_series.rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        ema_50 = ema_50_1w_aligned[i]
        ema_50_prev = ema_50_1w_aligned[i-1] if i > 0 else ema_50
        
        # Determine 1w EMA50 trend: rising if current > previous, falling if current < previous
        ema_50_rising = ema_50 > ema_50_prev
        ema_50_falling = ema_50 < ema_50_prev
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 1w EMA50 rising AND volume confirmation
            if price > upper and ema_50_rising and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND 1w EMA50 falling AND volume confirmation
            elif price < lower and ema_50_falling and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price reverts to middle or trend turns down
            if price < middle or not ema_50_rising or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price reverts to middle or trend turns up
            if price > middle or not ema_50_falling or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals