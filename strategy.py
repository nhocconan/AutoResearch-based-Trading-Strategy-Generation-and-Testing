#!/usr/bin/env python3
# 4h_donchian_breakout_volume_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and 12h trend filter.
# Long: Price breaks above Donchian(20) upper band with volume > 1.5x 20-period average and 12h close > 12h EMA50.
# Short: Price breaks below Donchian(20) lower band with volume > 1.5x 20-period average and 12h close < 12h EMA50.
# Exit: Price returns to Donchian midpoint or opposite band touch (whichever comes first).
# Uses 12h EMA50 for trend filter to avoid counter-trend trades in choppy markets.
# Target: 20-40 trades/year to minimize fee drag while capturing strong breakouts.
# Donchian breakouts work in both bull (trend continuation) and bear (sharp reversals) markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    close_12h_s = pd.Series(close_12h)
    ema_50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian/volume warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # 12h trend filter: close > EMA50 for uptrend, < EMA50 for downtrend
        trend_12h_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to midpoint OR touches lower band
            if close[i] <= donchian_mid[i] or close[i] <= donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to midpoint OR touches upper band
            if close[i] >= donchian_mid[i] or close[i] >= donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above upper band with volume and 12h uptrend
            if (close[i] > donchian_upper[i] and    # Break above upper band
                volume_confirmed and                # Volume spike
                trend_12h_up):                      # 12h uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower band with volume and 12h downtrend
            elif (close[i] < donchian_lower[i] and  # Break below lower band
                  volume_confirmed and              # Volume spike
                  trend_12h_down):                  # 12h downtrend
                position = -1
                signals[i] = -0.25
    
    return signals