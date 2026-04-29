#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-bar 6h high AND weekly close > weekly open (bullish weekly candle) AND volume > 1.5x 20-bar avg
# Short when price breaks below 20-bar 6h low AND weekly close < weekly open (bearish weekly candle) AND volume > 1.5x 20-bar avg
# Exit when price returns to 20-bar 6h midpoint (mean reversion to median)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Donchian channels provide clear breakout levels; weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation reduces false breakouts by requiring institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "6h_Donchian20_WeeklyTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels (20-bar period)
    if n < 20:
        return np.zeros(n)
    
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get weekly data for trend filter (weekly candle direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    # Weekly bullish = close > open, bearish = close < open
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    # Align weekly trend to 6h timeframe (wait for weekly candle to close)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Donchian levels
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        mid_band = donchian_mid[i]
        
        # Weekly trend
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price returns to midpoint (mean reversion)
            if curr_close <= mid_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint (mean reversion)
            if curr_close >= mid_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND weekly bullish AND volume confirmation
            if curr_high > upper_band and weekly_bull and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower band AND weekly bearish AND volume confirmation
            elif curr_low < lower_band and weekly_bear and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals