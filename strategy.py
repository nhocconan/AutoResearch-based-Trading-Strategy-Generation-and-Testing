#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema26_1d)
    trend_up = close > ema26_1d_aligned
    trend_down = close < ema26_1d_aligned
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (no look-ahead)
    # Since Ichimoku is calculated on 6h data, we need to align the components
    # but we'll use the current values directly as they are based on past data
    # For Senkou Span, we need to shift forward by 26 periods (but we'll handle this in logic)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day (4 * 6h)
    
    start_idx = 52  # Ensure Ichimoku calculation is valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema26_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TK cross above AND price above cloud AND 1d uptrend
            if (tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i] and 
                close[i] > cloud_top and trend_up[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TK cross below AND price below cloud AND 1d downtrend
            elif (tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i] and 
                  close[i] < cloud_bottom and trend_down[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TK cross below OR price below cloud OR trend turns down
            if (tenkan[i] < kijun[i] or close[i] < cloud_bottom or not trend_up[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross above OR price above cloud OR trend turns up
            if (tenkan[i] > kijun[i] or close[i] > cloud_top or not trend_down[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Cloud system with TK cross signals and 1d trend filter
# Long when Tenkan crosses above Kijun, price is above cloud, and 1d trend is up
# Short when Tenkan crosses below Kijun, price is below cloud, and 1d trend is down
# The cloud acts as dynamic support/resistance, reducing false signals
# Cooldown of 4 bars limits trades to ~20-50 per year. Position size 0.25 manages risk.
# Works in bull markets (captures uptrend continuations above cloud) and bear markets (captures downtrends below cloud)