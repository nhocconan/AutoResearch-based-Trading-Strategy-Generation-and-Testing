#!/usr/bin/env python3
# 6h_elder_ray_regime_volume_v2
# Hypothesis: 6h strategy using Elder Ray (Bull/Bear Power) with EMA200 trend filter and volume confirmation.
# Long: Bull Power > 0, close > EMA200, volume > 1.5x 20-period average.
# Short: Bear Power < 0, close < EMA200, volume > 1.5x 20-period average.
# Exit: Opposite Elder Ray signal or volume divergence.
# Uses 1d EMA200 for higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation filters weak breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA200 for trend filter (6h)
    close_s = pd.Series(close)
    ema200 = close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Elder Ray components (6h)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # 1d EMA200 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_s_1d = pd.Series(close_1d)
    ema200_1d = close_s_1d.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema200[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative OR volume divergence (price up but volume down)
            if bear_power[i] >= 0 or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR volume divergence (price down but volume down)
            if bull_power[i] <= 0 or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Bull Power positive, price above EMA200, price above 1d EMA200, volume confirmed
            if (bull_power[i] > 0 and close[i] > ema200[i] and close[i] > ema200_1d_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power negative, price below EMA200, price below 1d EMA200, volume confirmed
            elif (bear_power[i] < 0 and close[i] < ema200[i] and close[i] < ema200_1d_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals