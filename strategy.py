#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_Filter_v1
Hypothesis: 6h Ichimoku TK cross with Kumo twist (Senkou Span A/B crossover) as momentum signal,
filtered by 1-week trend (price > 1w EMA50 for long, < for short) and volume confirmation.
Ichimoku provides inherent trend/momentum/structure; weekly filter avoids counter-trend trades.
Volume spike confirms institutional participation. Designed for 12-37 trades/year (50-150 over 4 years)
by requiring confluence of TK cross, Kumo twist, weekly trend, and volume.
Works in bull/bear via 1-week trend filter: only takes longs in weekly uptrend, shorts in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for weekly trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Kumo twist: Senkou Span A crosses above/below Senkou Span B
    # We use the current values (not shifted) for twist detection, as the shift is for plotting
    # The twist itself is when Senkou A and Senkou B cross
    senkou_a_shifted = np.roll(senkou_a, 26)  # Actual Senkou A plotted 26 periods ahead
    senkou_b_shifted = np.roll(senkou_b, 26)  # Actual Senkou B plotted 26 periods ahead
    # But for twist detection, we check when the raw Senkou A/B cross (which determines future Kumo)
    kumou_twist_bull = senkou_a > senkou_b  # Senkou A above B -> future bullish Kumo
    kumou_twist_bear = senkou_a < senkou_b  # Senkou A below B -> future bearish Kumo
    
    # TK cross: Tenkan-sen crosses above/below Kijun-sen
    tk_cross_bull = tenkan > kijun
    tk_cross_bear = tenkan < kijun
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Ichimoku, 20 for volume MA, 50 for 1w EMA)
    start_idx = max(52, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Ichimoku signals
        # Long setup: TK cross bull + Kumo twist bull (Senkou A > B) + weekly uptrend
        long_setup = tk_cross_bull[i] and kumou_twist_bull[i] and weekly_trend[i] == 1 and volume_spike
        # Short setup: TK cross bear + Kumo twist bear (Senkou A < B) + weekly downtrend
        short_setup = tk_cross_bear[i] and kumou_twist_bear[i] and weekly_trend[i] == -1 and volume_spike
        
        # Exit conditions: reverse TK cross or Kumo twist reversal
        exit_long = tk_cross_bear[i] or (kumou_twist_bull[i] == False and senkou_a[i] < senkou_b[i])  # Twist turned bearish
        exit_short = tk_cross_bull[i] or (kumou_twist_bear[i] == False and senkou_a[i] > senkou_b[i])  # Twist turned bullish
        
        if long_setup and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_setup and position != -1:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0