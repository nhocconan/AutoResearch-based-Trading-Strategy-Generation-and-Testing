#!/usr/bin/env python3
# 12h_donchian_breakout_volume_v1
# Hypothesis: 12h strategy using Donchian channel breakouts with volume confirmation and 1d trend filter.
# Long: Price breaks above 20-period Donchian upper channel with volume > 1.5x 20-period average and 1d close > 1d EMA50.
# Short: Price breaks below 20-period Donchian lower channel with volume > 1.5x 20-period average and 1d close < 1d EMA50.
# Exit: Price returns to 20-period Donchian middle channel (mean of upper and lower).
# Uses 1d EMA50 for trend filter: only long when 1d close > 1d EMA50, only short when 1d close < 1d EMA50.
# Target: 12-30 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
# Donchian breakouts capture strong moves, volume confirmation reduces false breakouts, and 1d trend filter aligns with higher timeframe direction.
# Works in bull markets by catching breakouts and in bear markets by allowing short breakdowns with trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian and volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # 1d trend filter: close > EMA50 for uptrend, < EMA50 for downtrend
        trend_1d_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to middle channel
            if close[i] <= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to middle channel
            if close[i] >= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above upper channel with volume and 1d uptrend
            if (close[i] > donchian_upper[i] and    # Break above Donchian upper
                volume_confirmed and                # Volume spike
                trend_1d_up):                       # 1d uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower channel with volume and 1d downtrend
            elif (close[i] < donchian_lower[i] and  # Break below Donchian lower
                  volume_confirmed and              # Volume spike
                  trend_1d_down):                   # 1d downtrend
                position = -1
                signals[i] = -0.25
    
    return signals