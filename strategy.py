#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Breakout with 12h Trend Filter
# Uses Ichimoku system (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span) on 6h for entry signals
# 12h EMA50 provides trend filter to avoid counter-trend trades
# Long when price breaks above cloud AND Tenkan > Kijun (bullish TK cross) in uptrend (price > 12h EMA50)
# Short when price breaks below cloud AND Tenkan < Kijun (bearish TK cross) in downtrend (price < 12h EMA50)
# Designed for ~20-40 trades/year on 6h timeframe to minimize fee drag
# Ichimoku works well in both trending and ranging markets by providing dynamic support/resistance

name = "6h_Ichimoku_CloudBreakout_12hEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku calculations (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # For signal generation, we need current close vs close 26 periods ago
    close_lagged_26 = np.roll(close, 26)
    close_lagged_26[:26] = np.nan  # First 26 values are invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need Senkou Span B warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close_lagged_26[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_senkou_a = senkou_span_a[i]
        curr_senkou_b = senkou_span_b[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_close_lagged = close_lagged_26[i]
        
        # Cloud boundaries (Senkou Span A/B form the cloud)
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Bullish TK cross: Tenkan > Kijun
        tk_bullish = curr_tenkan > curr_kijun
        # Bearish TK cross: Tenkan < Kijun
        tk_bearish = curr_tenkan < curr_kijun
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Chikou confirmation: current close vs price 26 periods ago
        chikou_bullish = curr_close > curr_close_lagged
        chikou_bearish = curr_close < curr_close_lagged
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below cloud OR TK cross turns bearish
            if price_below_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above cloud OR TK cross turns bullish
            if price_above_cloud or not tk_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Trend filter: 12h EMA50
            uptrend = curr_close > curr_ema50_12h
            downtrend = curr_close < curr_ema50_12h
            
            # Long entry: bullish TK cross + price above cloud + chikou bullish + uptrend
            if tk_bullish and price_above_cloud and chikou_bullish and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TK cross + price below cloud + chikou bearish + downtrend
            elif tk_bearish and price_below_cloud and chikou_bearish and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals