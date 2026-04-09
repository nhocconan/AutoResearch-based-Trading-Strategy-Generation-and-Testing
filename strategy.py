#!/usr/bin/env python3
# 1d_donchian_breakout_weekly_volume_v1
# Hypothesis: 1d strategy using Donchian channel breakouts with weekly trend filter and volume confirmation.
# Long: price breaks above Donchian(20) high, close > weekly EMA50, volume > 1.5x 20-day average.
# Short: price breaks below Donchian(20) low, close < weekly EMA50, volume > 1.5x 20-day average.
# Exit: opposite Donchian breakout or volume divergence.
# Weekly trend filter avoids counter-trend trades. Volume confirmation filters weak breakouts.
# Target: 7-25 trades/year (30-100 total over 4 years) as per 1d timeframe limits.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_volume_v1"
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
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema50_1w = close_1w_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volume divergence (price up but volume down)
            if close[i] < donchian_low[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volume divergence (price down but volume down)
            if close[i] > donchian_high[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above Donchian high, close > weekly EMA50, volume confirmed
            if (close[i] > donchian_high[i] and close[i] > ema50_1w_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low, close < weekly EMA50, volume confirmed
            elif (close[i] < donchian_low[i] and close[i] < ema50_1w_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals