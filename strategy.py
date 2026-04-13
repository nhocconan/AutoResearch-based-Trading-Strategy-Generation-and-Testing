#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1d Ichimoku Cloud + TK Cross + Volume Confirmation
    # Long when price > Kumo cloud, Tenkan > Kijun, and volume spike
    # Short when price < Kumo cloud, Tenkan < Kijun, and volume spike
    # Exit when Tenkan/Kijun cross reverses or price re-enters cloud
    # Uses discrete size 0.25 to minimize fee churn. Target: 50-150 trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # For alignment, we'll use close shifted back 26 periods
    chikou = np.roll(close_1d, 26)
    chikou[:26] = np.nan  # First 26 values invalid
    
    # Calculate 1d volume mean (20-period) for volume confirmation
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Kumo Cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume filter: current 1d volume > 1.3 * 20-period mean (volume spike)
        volume_confirmation = volume_1d_aligned[i] > 1.3 * vol_ma_aligned[i]
        
        # Ichimoku signals:
        # Bullish: Price above cloud AND Tenkan > Kijun AND Chikou above price (26 periods ago)
        # Bearish: Price below cloud AND Tenkan < Kijun AND Chikou below price (26 periods ago)
        price = close[i]
        
        # For Chikou comparison, we need current price vs price 26 periods ago
        # Since Chikou is close plotted 26 periods behind, we compare:
        # Chikou > price 26 periods ago = bullish
        # Chikou < price 26 periods ago = bearish
        chikou_value = chikou_aligned[i]
        price_26_ago = close[i-26] if i >= 26 else np.nan
        
        if i < 26:
            chikou_bullish = False
            chikou_bearish = False
        else:
            chikou_bullish = chikou_value > price_26_ago
            chikou_bearish = chikou_value < price_26_ago
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        
        # Bullish conditions
        bullish = (price > upper_cloud and 
                  tenkan_val > kijun_val and 
                  chikou_bullish and 
                  volume_confirmation)
        
        # Bearish conditions
        bearish = (price < lower_cloud and 
                  tenkan_val < kijun_val and 
                  chikou_bearish and 
                  volume_confirmation)
        
        # Exit conditions: Tenkan/Kijun cross reverse or price re-enters cloud
        exit_long = (tenkan_val < kijun_val) or (price < upper_cloud and price > lower_cloud)
        exit_short = (tenkan_val > kijun_val) or (price < upper_cloud and price > lower_cloud)
        
        if bullish and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ichimoku_tk_cross_volume_v1"
timeframe = "6h"
leverage = 1.0