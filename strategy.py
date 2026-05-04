#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation
# Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B).
# Price above/below cloud indicates trend direction; TK cross signals momentum.
# 1w EMA50 filters for higher-timeframe trend to avoid counter-trend trades.
# Volume confirmation ensures breakouts have participation.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear markets via trend-filtered Ichimoku signals.

name = "6h_Ichimoku_Cloud_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Get 6h data for volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirmed = volume[i] > (1.3 * vol_ema_20[i])
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1w_aligned[i]
        
        # Ichimoku signals:
        # Bullish: Price above cloud AND Tenkan-sen > Kijun-sen (TK cross up)
        price_above_cloud = (close[i] > senkou_a[i]) and (close[i] > senkou_b[i])
        price_below_cloud = (close[i] < senkou_a[i]) and (close[i] < senkou_b[i])
        tk_cross_up = tenkan_sen[i] > kijun_sen[i]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i]
        
        if position == 0:
            # Long: Price above cloud + TK cross up + volume confirmation + bullish 1w trend
            if (price_above_cloud and tk_cross_up and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + TK cross down + volume confirmation + bearish 1w trend
            elif (price_below_cloud and tk_cross_down and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price falls below cloud OR TK cross down OR 1w trend turns bearish
            if (not price_above_cloud) or (not tk_cross_up) or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price rises above cloud OR TK cross up OR 1w trend turns bullish
            if (not price_below_cloud) or (not tk_cross_down) or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals