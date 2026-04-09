#!/usr/bin/env python3
# 1h_htf_ema_crossover_volume_v1
# Hypothesis: 1h strategy using 4h EMA crossover for trend direction, 1d EMA for regime filter, and volume spike for entry timing.
# Long: 4h EMA21 > EMA50, 1d close > EMA200, 1h volume > 2x 20-period average.
# Short: 4h EMA21 < EMA50, 1d close < EMA200, 1h volume > 2x 20-period average.
# Exit: Opposite 4h EMA crossover or volume drops below average.
# Uses discrete position sizing (0.20) to limit fee drag and control drawdown.
# Target: 15-37 trades/year (60-150 total over 4 years) by requiring confluence of HTF trend, regime, and volume.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_htf_ema_crossover_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for EMA crossover (trend direction)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    ema21_4h = close_4h_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50_4h = close_4h_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_crossover_4h = align_htf_to_ltf(prices, df_4h, ema21_4h - ema50_4h)  # >0 = bullish, <0 = bearish
    
    # Load 1d data ONCE for regime filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema200_1d = close_1d_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    close_gt_ema200_1d = align_htf_to_ltf(prices, df_1d, close_1d > ema200_1d)  # Boolean: True if above
    
    # 1h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_crossover_4h[i]) or np.isnan(close_gt_ema200_1d[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: 4h EMA crossover turns bearish OR volume drops below average
            if ema_crossover_4h[i] <= 0 or volume[i] <= volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 4h EMA crossover turns bullish OR volume drops below average
            if ema_crossover_4h[i] >= 0 or volume[i] <= volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: 4h bullish, 1d above EMA200, volume confirmed
            if (ema_crossover_4h[i] > 0 and close_gt_ema200_1d[i] and volume_confirmed):
                position = 1
                signals[i] = 0.20
            # Short entry: 4h bearish, 1d below EMA200, volume confirmed
            elif (ema_crossover_4h[i] < 0 and not close_gt_ema200_1d[i] and volume_confirmed):
                position = -1
                signals[i] = -0.20
    
    return signals