#!/usr/bin/env python3

name = "6h_Ichimoku_TK_Cross_CloudFilter_12hTrend"
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
    volume = prices['volume'].values
    
    # Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period)
    # Tenkan-sen = (9-period high + 9-period low) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    tenkan_sen = (high_series.rolling(window=9, min_periods=9).max() + 
                  low_series.rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    kijun_sen = (high_series.rolling(window=26, min_periods=26).max() + 
                 low_series.rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # TK Cross signals
    tk_cross_up = tenkan_sen > kijun_sen
    tk_cross_down = tenkan_sen < kijun_sen
    
    # Cloud components: Senkou Span A and B
    # Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B = (52-period high + 52-period low) / 2, shifted 26 periods ahead
    senkou_span_b = (high_series.rolling(window=52, min_periods=52).max() + 
                     low_series.rolling(window=52, min_periods=52).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Cloud top and bottom (without future shift for current price comparison)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Price above/below cloud (using current cloud values)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i]) or
            np.isnan(price_above_cloud[i]) or np.isnan(price_below_cloud[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross up + price above cloud + 12h uptrend
            if tk_cross_up[i] and price_above_cloud[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + 12h downtrend
            elif tk_cross_down[i] and price_below_cloud[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down OR price below cloud OR trend reversal
            if tk_cross_down[i] or not price_above_cloud[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up OR price above cloud OR trend reversal
            if tk_cross_up[i] or not price_below_cloud[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals