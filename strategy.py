#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation
    # Long: price breaks above Donchian upper band AND 12h EMA34 > previous 12h EMA34 (uptrend) AND volume > 1.5x 20-bar avg
    # Short: price breaks below Donchian lower band AND 12h EMA34 < previous 12h EMA34 (downtrend) AND volume > 1.5x 20-bar avg
    # Exit: price touches opposite Donchian band OR Donchian middle band (mean reversion)
    # Using 4h timeframe for optimal trade frequency (target 19-50/year), Donchian for structure,
    # 12h EMA for HTF trend filter (avoiding whipsaws), and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_band[i] = np.max(high[i-20:i])
        lower_band[i] = np.min(low[i-20:i])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = df_12h['close'].values
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema_12h[33] = np.mean(close_12h[:34])  # SMA for first value
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * multiplier + ema_12h[i-1]
    
    # Align 12h EMA34 to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h EMA34 slope (trend direction)
    ema_12h_slope = np.full_like(ema_12h_aligned, np.nan)
    for i in range(1, len(ema_12h_aligned)):
        if not np.isnan(ema_12h_aligned[i]) and not np.isnan(ema_12h_aligned[i-1]):
            ema_12h_slope[i] = ema_12h_aligned[i] - ema_12h_aligned[i-1]
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or
            np.isnan(ema_12h_slope[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_band[i]
        breakout_lower = close[i] < lower_band[i]
        
        # Exit conditions: touch opposite band or middle band
        touch_lower = close[i] < lower_band[i]
        touch_upper = close[i] > upper_band[i]
        touch_middle = abs(close[i] - middle_band[i]) < (upper_band[i] - lower_band[i]) * 0.05  # Within 5% of middle
        
        # Trend filter: 12h EMA34 slope
        uptrend = ema_12h_slope[i] > 0
        downtrend = ema_12h_slope[i] < 0
        
        # Entry logic: Donchian breakout + trend filter + volume confirmation
        long_entry = breakout_upper and uptrend and volume_spike[i]
        short_entry = breakout_lower and downtrend and volume_spike[i]
        
        # Exit logic: opposite band touch or middle band reversion
        long_exit = touch_lower or touch_middle
        short_exit = touch_upper or touch_middle
        
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