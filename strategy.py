# 1
#!/usr/bin/env python3

name = "6h_Ichimoku_Kumo_Twist_Filter"
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
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo twist: Senkou Span A crosses above/below Senkou Span B
    # Kumo twist bullish: Senkou Span A > Senkou Span B
    # Kumo twist bearish: Senkou Span A < Senkou Span B
    kumo_twist_bullish = senkou_span_a_6h > senkou_span_b_6h
    kumo_twist_bearish = senkou_span_a_6h < senkou_span_b_6h
    
    # TK Cross: Tenkan-sen crosses Kijun-sen
    tk_cross_bullish = tenkan_sen_6h > kijun_sen_6h
    tk_cross_bearish = tenkan_sen_6h < kijun_sen_6h
    
    # Price relative to Kumo (cloud)
    price_above_kumo = (close > senkou_span_a_6h) & (close > senkou_span_b_6h)
    price_below_kumo = (close < senkou_span_a_6h) & (close < senkou_span_b_6h)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~3 days for 6h to reduce trades
    
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_6h[i]) or 
            np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or 
            np.isnan(senkou_span_b_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Kumo twist bullish + TK cross bullish + price above Kumo + volume
            if (kumo_twist_bullish[i] and 
                tk_cross_bullish[i] and 
                price_above_kumo[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Kumo twist bearish + TK cross bearish + price below Kumo + volume
            elif (kumo_twist_bearish[i] and 
                  tk_cross_bearish[i] and 
                  price_below_kumo[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Kumo twist bearish or TK cross bearish or price drops below Kumo
            if (not kumo_twist_bullish[i]) or (not tk_cross_bullish[i]) or (not price_above_kumo[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Kumo twist bullish or TK cross bullish or price rises above Kumo
            if (not kumo_twist_bearish[i]) or (not tk_cross_bearish[i]) or (not price_below_kumo[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) indicates trend acceleration, 
# TK cross confirms momentum alignment, and price position relative to cloud filters 
# for trend strength. This strategy works in both bull and bear markets by capturing 
# strong trend continuations. The 6h timeframe reduces noise while Kumo twist provides 
# early trend signals. Volume confirmation avoids false signals, cooldown reduces 
# trade frequency to ~15-30 trades/year. Kumo twist is a leading indicator of trend 
# changes, making it effective for catching new trends early in both directions.