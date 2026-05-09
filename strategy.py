#!/usr/bin/env python3
"""
6h_FisherTransform_12hTrend_Volume
Hypothesis: Ehlers Fisher Transform identifies turning points in price cycles. 
Long when Fisher crosses above -1.5 with 12h uptrend and volume confirmation.
Short when Fisher crosses below +1.5 with 12h downtrend and volume confirmation.
Works in bull/bear markets by following 12h trend and using Fisher for precise timing.
Designed for low trade frequency (15-35/year) with high win rate by requiring trend alignment,
volume confirmation, and Fisher reversal signals.
"""

name = "6h_FisherTransform_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ehlers Fisher Transform (9-period)
    hl_range = high - low
    hlc3 = (high + low + close) / 3
    max_hl = pd.Series(hl_range).rolling(window=9, min_periods=9).max().values
    min_hl = pd.Series(hl_range).rolling(window=9, min_periods=9).min().values
    # Avoid division by zero
    denom = max_hl - min_hl
    denom = np.where(denom == 0, 1e-10, denom)
    value1 = 0.66 * ((hlc3 - min_hl) / denom - 0.5) + 0.67 * np.roll(0.66 * ((hlc3 - min_hl) / denom - 0.5) + 0.67 * np.zeros_like(hlc3), 1)
    value1 = np.where(np.arange(len(value1)) < 1, 0, value1)
    value1 = np.clip(value1, -0.999, 0.999)
    fish = 0.5 * np.log((1 + value1) / (1 - value1)) + 0.5 * np.roll(0.5 * np.log((1 + value1) / (1 - value1)) + 0.5 * np.zeros_like(value1), 1)
    fish = np.where(np.arange(len(fish)) < 1, 0, fish)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA34 trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    trend_up = close > ema_34_12h_aligned
    trend_down = close < ema_34_12h_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(fish[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 + 12h uptrend + volume spike + session
            if fish[i] > -1.5 and fish[i-1] <= -1.5 and trend_up[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 + 12h downtrend + volume spike + session
            elif fish[i] < 1.5 and fish[i-1] >= 1.5 and trend_down[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below 0 or trend reversal
            if fish[i] < 0 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above 0 or trend reversal
            if fish[i] > 0 or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals