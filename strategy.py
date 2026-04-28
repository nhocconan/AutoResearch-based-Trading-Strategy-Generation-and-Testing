#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Enter long when price breaks above Donchian(20) upper band, weekly EMA50 is rising, and volume > 1.5x 20-bar average.
# Enter short when price breaks below Donchian(20) lower band under same conditions.
# Exit when price crosses Donchian midpoint.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
# Weekly trend filter ensures we trade with the higher timeframe momentum, reducing whipsaws.
# Volume confirmation adds conviction to breakouts.

name = "1d_DonchianBreakout_1wEMA50Trend_VolumeConfirm_v1"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_prev = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().shift(1).values
    ema_50_1w_rising = ema_50_1w > ema_50_1w_prev
    
    # Align weekly EMA trend to daily timeframe
    ema_50_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_rising)
    
    # Donchian channels (20-period) on 1d
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_rising_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Weekly trend filter: EMA50 rising = bullish trend
        weekly_bullish = ema_50_1w_rising_aligned[i]
        weekly_bearish = ~weekly_bullish
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]  # Break below previous period's low
        
        # Exit conditions
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        # Handle entries and exits
        if breakout_up and weekly_bullish and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and weekly_bearish and vol_confirm and position >= 0:
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