#!/usr/bin/env python3
# 6h_12h_ichimoku_cloud_trend_v1
# Hypothesis: 6h strategies based on 12h Ichimoku cloud with 1d trend filter work in both bull and bear markets.
# Long: price above 12h Ichimoku cloud AND 1d close > 1d EMA50
# Short: price below 12h Ichimoku cloud AND 1d close < 1d EMA50
# Exit: price crosses the 12h Tenkan-sen/Kijun-sen midpoint (TK cross)
# Uses 6h primary timeframe with 12h HTF for Ichimoku and 1d HTF for EMA50 trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # need at least 52 bars for Ichimoku
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Ichimoku components on 12h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Shift will be handled by align_htf_to_ltf with additional_delay_bars
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    # Shift will be handled by align_htf_to_ltf with additional_delay_bars
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Ichimoku components to 6h timeframe
    # Tenkan-sen and Kijun-sen are contemporaneous, so no extra delay needed beyond bar close
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    
    # Senkou Span A and B are leading indicators, need 26-period shift for proper alignment
    # align_htf_to_ltf already waits for the HTF bar to close, so we add 26 bars delay
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b, additional_delay_bars=26)
    
    # Align 1d EMA50 to 6h timeframe (no extra delay needed for EMA)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # warmup for Ichimoku calculations
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross for exit signal
        tk_cross = (tenkan_aligned[i] - kijun_aligned[i]) * (tenkan_aligned[i-1] - kijun_aligned[i-1]) < 0
        
        if position == 1:  # Long position
            # Exit: price crosses below cloud OR TK cross (tenkan crosses below kijun)
            if price < cloud_bottom or tk_cross:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above cloud OR TK cross (tenkan crosses above kijun)
            if price > cloud_top or tk_cross:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price above cloud AND 1d EMA50 uptrend (close > EMA50)
            if price > cloud_top and close_1d[-1] > ema_50_1d[-1] if len(close_1d) > 0 and len(ema_50_1d) > 0 else False:
                # Check if we have valid 1d data for current bar
                # Find the corresponding 1d bar index for current 6h bar
                # Since we aligned ema_50_aligned, we can use it directly
                if not np.isnan(ema_50_aligned[i]) and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: price below cloud AND 1d EMA50 downtrend (close < EMA50)
            elif price < cloud_bottom and not np.isnan(ema_50_aligned[i]) and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals