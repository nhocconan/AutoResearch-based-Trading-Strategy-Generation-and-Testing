#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation
    # Long: price breaks above upper band AND 12h EMA34 > price (uptrend) AND volume > 1.3x avg
    # Short: price breaks below lower band AND 12h EMA34 < price (downtrend) AND volume > 1.3x avg
    # Exit: price touches opposite band or retests breakout level
    # Using 4h timeframe for optimal trade frequency (target 19-50/year), Donchian for structure,
    # 12h EMA34 for trend filter, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34)
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] * multiplier) + (ema_34_12h[i-1] * (1 - multiplier))
    
    # Align 12h EMA to 4h
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h Donchian channels (20-period)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_band[i] = np.max(high[i-20:i])
        lower_band[i] = np.min(low[i-20:i])
    
    # Get 4h volume for confirmation (>1.3x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA34 > price = uptrend, EMA34 < price = downtrend
        uptrend = ema_34_12h_aligned[i] > close[i]
        downtrend = ema_34_12h_aligned[i] < close[i]
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_band[i]
        breakout_lower = close[i] < lower_band[i]
        
        # Exit conditions: touch opposite band or retest breakout level
        touch_lower = close[i] < lower_band[i]  # Exit long on lower band touch
        touch_upper = close[i] > upper_band[i]  # Exit short on upper band touch
        retest_upper = close[i] < upper_band[i] and position == 1  # Long exit on upper band retest
        retest_lower = close[i] > lower_band[i] and position == -1  # Short exit on lower band retest
        
        # Entry logic: Donchian breakout + trend filter + volume confirmation
        long_entry = breakout_upper and uptrend and volume_spike[i]
        short_entry = breakout_lower and downtrend and volume_spike[i]
        
        # Exit logic: opposite band touch or breakout level retest
        long_exit = touch_lower or retest_upper
        short_exit = touch_upper or retest_lower
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_ema34_volume_v1"
timeframe = "4h"
leverage = 1.0