#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_regime_v1
# Hypothesis: Daily Donchian channel breakout with weekly trend filter and volume confirmation.
# Long: Price breaks above 20-day Donchian high AND weekly close > weekly EMA50 AND volume > 1.5x 20-day avg volume
# Short: Price breaks below 20-day Donchian low AND weekly close < weekly EMA50 AND volume > 1.5x 20-day avg volume
# Exit: Opposite Donchian breakout or price crosses 10-day EMA in opposite direction
# Uses 1d primary timeframe with 1w HTF for trend filter.
# Target: 50-100 total trades over 4 years to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_regime_v1"
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
    
    # Calculate 20-day Donchian channels with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day EMA for exit with min_periods
    close_series = pd.Series(close)
    ema10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-day average volume for confirmation with min_periods
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA50 on 1w with min_periods
    close_1w_s = pd.Series(close_1w)
    ema50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema10[i]) or np.isnan(vol_ma20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(close[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price breaks below 20-day Donchian low (opposite breakout)
            # 2. Price crosses below 10-day EMA (trend weakening)
            if low[i] < donchian_low[i] or close[i] < ema10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price breaks above 20-day Donchian high (opposite breakout)
            # 2. Price crosses above 10-day EMA (trend weakening)
            if high[i] > donchian_high[i] or close[i] > ema10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above 20-day Donchian high AND weekly close > weekly EMA50 AND volume confirmed
            if high[i] > donchian_high[i] and close[i] > ema50_1w_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below 20-day Donchian low AND weekly close < weekly EMA50 AND volume confirmed
            elif low[i] < donchian_low[i] and close[i] < ema50_1w_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals