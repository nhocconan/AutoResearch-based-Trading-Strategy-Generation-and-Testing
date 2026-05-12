#/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_Volume
# Hypothesis: Ichimoku Tenkan-Kijun cross with cloud filter from daily timeframe and volume confirmation.
# Tenkan-sen (9-period) crossing above Kijun-sen (26-period) signals momentum shift.
# Price above/below daily Kumo (cloud) determines trend direction for filtering trades.
# Volume > 1.5x 20-period SMA confirms breakout strength.
# Works in both bull/bear markets: cloud acts as dynamic support/resistance, TK cross captures momentum shifts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Ichimoku cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Ichimoku components on daily data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Actual cloud is plotted 26 periods ahead, so we need to shift back for current price comparison
    kumou_top = np.maximum(senkou_a_1d, senkou_b_1d)  # Upper cloud boundary
    kumou_bottom = np.minimum(senkou_a_1d, senkou_b_1d)  # Lower cloud boundary

    # Calculate Ichimoku on 6h data for TK cross signals
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    
    # TK cross signals: Tenkan crossing above/below Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h.shift(1) <= kijun_6h.shift(1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h.shift(1) >= kijun_6h.shift(1))

    # Align daily Ichimoku components to 6h timeframe
    kumou_top_aligned = align_htf_to_ltf(prices, df_1d, kumou_top)
    kumou_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumou_bottom)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(kumou_top_aligned[i]) or np.isnan(kumou_bottom_aligned[i]) or
            np.isnan(volume_sma20[i]) or np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check for TK cross at current bar
        current_tk_up = tk_cross_up[i] if i < len(tk_cross_up) else False
        current_tk_down = tk_cross_down[i] if i < len(tk_cross_down) else False

        if position == 0:
            # LONG: TK cross up + price above cloud + volume confirmation
            if (current_tk_up and
                close[i] > kumou_top_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross down + price below cloud + volume confirmation
            elif (current_tk_down and
                  close[i] < kumou_bottom_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross down OR price falls below cloud
            if (current_tk_down or close[i] < kumou_bottom_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross up OR price rises above cloud
            if (current_tk_up or close[i] > kumou_top_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals