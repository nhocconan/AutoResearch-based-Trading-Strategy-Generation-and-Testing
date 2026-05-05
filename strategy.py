#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND price > 1d EMA50 (bullish regime) AND volume > 1.5x 20-period average
# Short when price breaks below Kumo (cloud) AND Tenkan < Kijun (bearish TK cross) AND price < 1d EMA50 (bearish regime) AND volume > 1.5x 20-period average
# Exit when price re-enters Kumo OR TK cross reverses
# Uses 6h primary timeframe with 1d HTF for trend filter to reduce whipsaw and avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Ichimoku provides dynamic support/resistance; TK cross confirms momentum; volume filter ensures conviction

name = "6h_Ichimoku_TK_Cross_Cloud_Breakout_1dEMA50_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # TK cross: Tenkan > Kijun (bullish) or Tenkan < Kijun (bearish)
    tk_bullish = tenkan > kijun
    tk_bearish = tenkan < kijun
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any value is NaN
        if (np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tk_bullish[i]) or 
            np.isnan(tk_bearish[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper cloud AND bullish TK cross AND price > 1d EMA50 AND volume spike
            if (close[i] > upper_cloud[i] and 
                tk_bullish[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower cloud AND bearish TK cross AND price < 1d EMA50 AND volume spike
            elif (close[i] < lower_cloud[i] and 
                  tk_bearish[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud OR TK cross turns bearish
            if close[i] <= upper_cloud[i] or not tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters cloud OR TK cross turns bullish
            if close[i] >= lower_cloud[i] or not tk_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals