#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Use Ichimoku (Tenkan/Kijun/Senkou) from 6h for entry signals
# - Only take trades in direction of 1d trend (price above/below 1d Kumo)
# - Require volume > 1.5x average for confirmation
# - Exit when price crosses opposite Kumo edge or TK cross reverses
# - Target: 60-150 total trades over 4 years (15-38/year)
# - Works in bull/bear: Cloud acts as dynamic support/resistance, trend filter avoids counter-trend

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (Kumo)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for signals but could be used for confirmation
    
    # Align Ichimoku components to current 6h bars (shifted for look-ahead prevention)
    # Senkou spans are already forward-shifted, so we align then use current values
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(tenkan))}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(kijun))}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(senkou_a))}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(senkou_b))}), senkou_b)
    
    # 1d Kumo (cloud) for trend filter
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan and Kijun
    max_high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_9_1d + min_low_9_1d) / 2
    
    max_high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_26_1d + min_low_26_1d) / 2
    
    # 1d Senkou Span A and B
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    max_high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((max_high_52_1d + min_low_52_1d) / 2)
    
    # Align 1d Kumo to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # The Kumo/cloud is between Senkou A and Senkou B
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    lower_cloud_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Volume confirmation: 1.5x average volume
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup for Ichimoku
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(upper_cloud_1d[i]) or np.isnan(lower_cloud_1d[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine Kumo boundaries for 6h (current cloud)
        upper_cloud_6h = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud_6h = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross signals
        tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Kumo OR TK cross turns bearish
            elif close[i] < lower_cloud_6h or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Kumo OR TK cross turns bullish
            elif close[i] > upper_cloud_6h or not tk_cross_bearish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross in direction of 1d trend with volume confirmation
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            
            # 1d trend filter: price relative to 1d Kumo
            price_above_1d_kumo = close[i] > upper_cloud_1d[i]
            price_below_1d_kumo = close[i] < lower_cloud_1d[i]
            
            # Long: bullish TK cross + price above 1d Kumo (uptrend) + volume spike
            if tk_cross_bullish and price_above_1d_kumo and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish TK cross + price below 1d Kumo (downtrend) + volume spike
            elif tk_cross_bearish and price_below_1d_kumo and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals