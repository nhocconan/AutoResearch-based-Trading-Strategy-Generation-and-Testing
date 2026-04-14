#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Weekly Trend Filter and Volume Spike
# Uses weekly Donchian channels (20-period) for volatility-based breakout entries
# Weekly trend filter (EMA 50) ensures trading in direction of higher timeframe trend
# Volume confirmation (>2x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of weekly trend
# Target: 10-30 trades/year (40-120 total over 4 years) to minimize fee drift

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().values
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA 50
    close_series = pd.Series(close_1w)
    ema_50_1w = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume_1w)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Align weekly indicators to daily timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1w, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1w, lower_donchian)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    avg_vol_aligned = align_htf_to_ltf(prices, df_1w, avg_vol)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for weekly EMA and Donchian calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or np.isnan(avg_vol_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of weekly EMA
        trend_up = price > ema_50_1w_aligned[i]
        trend_down = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly upper Donchian with volume filter and uptrend
            if price > upper_donchian_aligned[i] and vol > 2.0 * avg_vol_aligned[i] and trend_up:
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly lower Donchian with volume filter and downtrend
            elif price < lower_donchian_aligned[i] and vol > 2.0 * avg_vol_aligned[i] and trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly lower Donchian (mean reversion)
            if price < lower_donchian_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above weekly upper Donchian (mean reversion)
            if price > upper_donchian_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyDonchian_Breakout_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0