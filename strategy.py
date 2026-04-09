#!/usr/bin/env python3
# 1h_mtf_volume_price_action_v1
# Hypothesis: 1h strategy using 4h/1d HTF for signal direction and 1h price action for entry timing.
# HTF Direction: 4h EMA(50) > 1d EMA(200) = bullish bias, < = bearish bias.
# Entry: 1h pullback to 4h EMA(20) with volume confirmation (>1.5x 20-bar average volume).
# Exit: Opposite HTF direction signal or price breaks 1h EMA(5) in opposite direction.
# Session filter: 08-20 UTC to avoid low-volume Asian session noise.
# Discrete sizing: ±0.20 to minimize fee churn. Target: 15-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_mtf_volume_price_action_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for EMA(50) - medium-term trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d HTF data for EMA(200) - long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h indicators for entry timing and exit
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_5_1h = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    open_time = prices['open_time']
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(ema_20_1h[i]) or np.isnan(ema_5_1h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine HTF bias: 4h EMA50 vs 1d EMA200
        htf_bullish = ema_50_4h_aligned[i] > ema_200_1d_aligned[i]
        htf_bearish = ema_50_4h_aligned[i] < ema_200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: HTF turns bearish OR price breaks below 1h EMA5
            if htf_bearish or close[i] < ema_5_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: HTF turns bullish OR price breaks above 1h EMA5
            if htf_bullish or close[i] > ema_5_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Volume confirmation: current volume > 1.5x 20-bar average
            volume_confirmed = volume[i] > 1.5 * volume_ma_20[i]
            
            if volume_confirmed:
                # Long entry: HTF bullish AND price > 1h EMA20 (pullback buy)
                if htf_bullish and close[i] > ema_20_1h[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: HTF bearish AND price < 1h EMA20 (pullback sell)
                elif htf_bearish and close[i] < ema_20_1h[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals