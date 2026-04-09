#!/usr/bin/env python3
# 4h_donchian_12h_trend_volume_v1
# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Works in bull/bear: 12h EMA20 defines medium-term trend; Donchian(20) breakout captures
# momentum; volume confirms institutional participation. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend and Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need sufficient data for EMA20 and Donchian
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA20 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # 12h Donchian channels (20-period)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (completed 12h bar only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian Lower OR trend turns bearish
            if close[i] < donchian_lower_aligned[i] or close[i] < ema20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian Upper OR trend turns bullish
            if close[i] > donchian_upper_aligned[i] or close[i] > ema20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above Donchian Upper with bullish trend
                if close[i] > donchian_upper_aligned[i] and close[i] > ema20_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian Lower with bearish trend
                elif close[i] < donchian_lower_aligned[i] and close[i] < ema20_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals