#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku
        return signals
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = close_1d.copy()  # Will be used for confirmation
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Additional filters: volume confirmation and trend strength
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(52, n):  # Start after Ichimoku calculation period
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Ichimoku conditions
        # Cloud (Kumo) is between Senkou Span A and B
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK Cross: Tenkan crosses Kijun
        tk_cross_up = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_cross_down = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        # Price above/below cloud
        price_above_cloud = price_close > cloud_top
        price_below_cloud = price_close < cloud_bottom
        
        # Chikou confirmation: Chikou (current close) vs price 26 periods ago
        chikou_conf_long = chikou_aligned[i] > close[i-26] if i >= 26 else False
        chikou_conf_short = chikou_aligned[i] < close[i-26] if i >= 26 else False
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: TK cross up + price above cloud + Chikou confirms + volume
        if tk_cross_up and price_above_cloud and chikou_conf_long and vol_confirm:
            enter_long = True
        
        # Short: TK cross down + price below cloud + Chikou confirms + volume
        if tk_cross_down and price_below_cloud and chikou_conf_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite TK cross or price crosses opposite Kijun
        exit_long = tk_cross_down or (price_close < kijun_aligned[i])
        exit_short = tk_cross_up or (price_close > kijun_aligned[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Ichimoku trend following strategy using daily Ichimoku cloud for trend direction.
# Enters long when Tenkan crosses above Kijun (TK cross up), price is above cloud,
# Chikou span confirms bullish momentum, and volume is above average.
# Enters short when Tenkan crosses below Kijun (TK cross down), price is below cloud,
# Chikou span confirms bearish momentum, and volume is above average.
# Exits on opposite TK cross or when price crosses Kijun line.
# Uses Ichimoku's built-in multi-timeframe nature (daily cloud for 6s entries) to filter noise.
# Works in both bull and bear markets by following the dominant trend on higher timeframe.
# Position size 0.25 to manage risk, targeting 15-35 trades per year (60-140 total over 4 years).
# Ichimoku is a proven trend-following system that works well in crypto markets.