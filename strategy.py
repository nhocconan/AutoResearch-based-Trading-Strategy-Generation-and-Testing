#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_v2
# Hypothesis: Ichimoku Cloud on daily timeframe for trend direction, with 6h Tenkan-Kijun cross for entry timing.
# Uses Kumo (cloud) from daily to filter trend, and Tenkan/Kijun cross on 6h for entry. Exit when price exits cloud.
# Works in bull/bear as cloud adapts to volatility. Targets 60-120 total trades over 4 years (15-30/year).
# Includes volume confirmation to avoid false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_v2"
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
    
    # Calculate ATR(20) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.full(n, np.nan)
    if n >= 20:
        atr[19] = np.mean(tr[:20])
        for i in range(20, n):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Load 1d data ONCE before loop for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Senkou B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku calculations (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2.0)
    
    # Chikou Span (Lagging Span): Current close shifted 26 periods behind
    # Not used for signals but calculated for completeness
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ok = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup for Ichimoku
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.03 * close[i]  # ATR less than 3% of price
        
        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        lower_cloud = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        in_cloud = (close[i] >= lower_cloud) and (close[i] <= upper_cloud)
        above_cloud = close[i] > upper_cloud
        below_cloud = close[i] < lower_cloud
        
        # Trend is bullish when Senkou A > Senkou B
        bullish_trend = senkou_a_6h[i] > senkou_b_6h[i]
        bearish_trend = senkou_a_6h[i] < senkou_b_6h[i]
        
        if position == 1:  # Long position
            # Exit: price closes back into or below cloud (or trend turns bearish)
            if in_cloud or below_cloud or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back into or above cloud (or trend turns bullish)
            if in_cloud or above_cloud or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud, bullish trend, TK cross up, volume confirmation
            if (above_cloud and bullish_trend and 
                tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and
                vol_ok and vol_filter):
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud, bearish trend, TK cross down, volume confirmation
            elif (below_cloud and bearish_trend and 
                  tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and
                  vol_ok and vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals