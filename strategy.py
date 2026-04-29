#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter
# Long when: price breaks above 4h Donchian(20) upper band AND volume > 1.5x 20-bar avg AND close > 1d EMA50
# Short when: price breaks below 4h Donchian(20) lower band AND volume > 1.5x 20-bar avg AND close < 1d EMA50
# Exit: reverse signal or volume drops below average
# Uses 4h for structure/direction, 1h for precise entry timing, 1d for trend filter
# Position size: 0.20 (discrete to minimize fee churn)
# Target: 15-37 trades/year on 1h (60-150 over 4 years)

name = "1h_DonchianBreakout_4hStructure_1dTrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (structure/direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume (on 1h)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Session filter: 08:00-20:00 UTC (reduces noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need sufficient history for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i] and session_filter[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when: price breaks above 4h Donchian upper AND volume confirmation AND close > 1d EMA50
            if curr_high > donchian_upper_4h_aligned[i] and vol_conf and curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short when: price breaks below 4h Donchian lower AND volume confirmation AND close < 1d EMA50
            elif curr_low < donchian_lower_4h_aligned[i] and vol_conf and curr_close < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on reverse signal or volume filter fail
            if (curr_low < donchian_lower_4h_aligned[i] and vol_conf) or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on reverse signal or volume filter fail
            if (curr_high > donchian_upper_4h_aligned[i] and vol_conf) or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals