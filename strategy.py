#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND 1w EMA(50) rising AND volume > 1.5x 20-day average volume
# Short when price breaks below Donchian(20) lower band AND 1w EMA(50) falling AND volume > 1.5x 20-day average volume
# Exit when price crosses back through the Donchian(20) midline (average of upper and lower bands)
# Designed for 1d timeframe with low trade frequency (target: 10-25/year) to avoid fee drag.
# Uses 1w for trend direction to ensure we only trade with the higher timeframe trend.
# Volume confirmation ensures breakouts have conviction, reducing false breakouts.
# Works in bull markets via long breakouts in uptrend, in bear markets via short breakdowns in downtrend.
name = "1d_DonchianBreakout_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    upper_band = np.full_like(high, np.nan, dtype=float)
    lower_band = np.full_like(low, np.nan, dtype=float)
    mid_band = np.full_like(close, np.nan, dtype=float)
    
    for i in range(lookback-1, n):
        upper_band[i] = np.max(high[i-lookback+1:i+1])
        lower_band[i] = np.min(low[i-lookback+1:i+1])
        mid_band[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    # 20-day average volume for confirmation
    vol_ma20 = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_rising[1:] = ema_50_1w[1:] > ema_50_1w[:-1]
    ema_50_falling[1:] = ema_50_1w[1:] < ema_50_1w[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(49, 19)  # Need 50 for 1w EMA and 20 for Donchian/volume
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(mid_band[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band AND 1w EMA50 rising AND volume > 1.5x average
            long_condition = (close[i] > upper_band[i]) and ema_50_rising_aligned[i] and (volume[i] > 1.5 * vol_ma20[i])
            # Short: price breaks below lower band AND 1w EMA50 falling AND volume > 1.5x average
            short_condition = (close[i] < lower_band[i]) and ema_50_falling_aligned[i] and (volume[i] > 1.5 * vol_ma20[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below midline
            if close[i] < mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above midline
            if close[i] > mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals