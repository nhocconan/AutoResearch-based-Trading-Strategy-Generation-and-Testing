#!/usr/bin/env python3
# 6H_Ichimoku_KumoBreakout_1wTrend_Filter
# Hypothesis: Ichimoku Kumo breakout on 6h chart with weekly trend filter for trend-following entries.
# Uses Kumo (Senkou Span A/B) as dynamic support/resistance and TK cross for entry timing.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed to work in bull markets (breakouts above Kumo) and bear markets (breakdowns below Kumo).
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6H_Ichimoku_KumoBreakout_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1w Data (loaded ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Weekly EMA50 Trend Filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Ichimoku Components (9, 26, 52) ===
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud): Senkou Span A and B shifted 26 periods ahead
    # For backtesting, we use the current cloud (no look-ahead)
    # Kumo top = max(senkou_a, senkou_b)
    # Kumo bottom = min(senkou_a, senkou_b)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Ichimoku calculations)
    start_idx = 52  # Need 52 periods for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(ema50_1w_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Kumo + TK cross bullish + weekly uptrend
            if (close[i] > kumo_top[i] and 
                tenkan[i] > kijun[i] and 
                close[i] > ema50_1w_6h[i]):
                signals[i] = position_size
                position = 1
            # Short: Price breaks below Kumo + TK cross bearish + weekly downtrend
            elif (close[i] < kumo_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  close[i] < ema50_1w_6h[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: TK cross in opposite direction OR price re-enters Kumo
            if position == 1:
                if (tenkan[i] < kijun[i]) or (close[i] < kumo_top[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if (tenkan[i] > kijun[i]) or (close[i] > kumo_bottom[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals