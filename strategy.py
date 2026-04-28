#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above upper Donchian channel with 1d EMA50 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below lower Donchian channel with 1d EMA50 downtrend and volume > 1.5x 20-bar average.
# Exit when price retreats to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 80-150 total trades over 4 years (20-38/year).
# Donchian channels provide robust trend-following structure; 1d EMA50 ensures higher timeframe alignment;
# volume confirmation filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Donchian20_EMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    channel_mid = (upper_channel + lower_channel) / 2
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(channel_mid[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > upper channel, EMA50 up, volume confirm
            if price > upper_channel[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < lower channel, EMA50 down, volume confirm
            elif price < lower_channel[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at channel midpoint
            if price <= channel_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at channel midpoint
            if price >= channel_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals