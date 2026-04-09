#!/usr/bin/env python3
# 12h_donchian_breakout_volume_v1
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
# Long: Price breaks above 20-period high + volume > 1.5x 20-period avg + close > 1d EMA50
# Short: Price breaks below 20-period low + volume > 1.5x 20-period avg + close < 1d EMA50
# Exit: Opposite breakout or volume divergence.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.

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
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA50 for trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_s_1d = pd.Series(close_1d)
    ema50_1d = close_s_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR volume divergence (price up but volume down)
            if close[i] < donchian_low[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR volume divergence (price down but volume down)
            if close[i] > donchian_high[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high + volume confirmed + 1d uptrend
            if (close[i] > donchian_high[i] and volume_confirmed and close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + volume confirmed + 1d downtrend
            elif (close[i] < donchian_low[i] and volume_confirmed and close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals