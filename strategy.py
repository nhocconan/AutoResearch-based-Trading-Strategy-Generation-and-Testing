#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Enter long when price breaks above Donchian upper band (20-period high), 12h EMA50 trending up, and volume > 2.0x 20-bar average.
# Enter short when price breaks below Donchian lower band (20-period low), 12h EMA50 trending down, and volume > 2.0x 20-bar average.
# Exit when price reaches opposite Donchian band or crosses the 12h EMA50.
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid excessive fee drag.
# Donchian channels provide clear trend-following structure; EMA50 filters for 12h trend alignment; volume confirms breakout strength.

name = "6h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels (20-period) from 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        # Exit conditions: price reaches opposite Donchian band or crosses 12h EMA50
        exit_long = close[i] < donchian_lower[i] or close[i] < ema_50_aligned[i]
        exit_short = close[i] > donchian_upper[i] or close[i] > ema_50_aligned[i]
        
        # Handle entries and exits
        if breakout_up and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals