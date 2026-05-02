#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses Tenkan-sen (9-period) and Kijun-sen (26-period) cross for entry signals
# Cloud (Senkou Span A/B) acts as dynamic support/resistance - price must be above/below cloud
# 1d EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>1.5 x 20-period EMA) confirms breakout validity
# Works in bull markets (price above cloud + bullish TK cross + 1d uptrend) and bear markets (price below cloud + bearish TK cross + 1d downtrend)
# Discrete position sizing (0.25) minimizes fee churn and controls drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_Ichimoku_Cloud_TK_Cross_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Ichimoku calculation)
    start_idx = 52
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = min(senkou_span_a[i], senkou_span_b[i])
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # TK cross signals
        tk_cross_bullish = tenkan_sen[i] > kijun_sen[i]
        tk_cross_bearish = tenkan_sen[i] < kijun_sen[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price above cloud + bullish TK cross + volume confirmation + uptrend
            if (close[i] > upper_cloud and tk_cross_bullish and 
                volume_confirmation[i] and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + bearish TK cross + volume confirmation + downtrend
            elif (close[i] < lower_cloud and tk_cross_bearish and 
                  volume_confirmation[i] and downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price falls below cloud OR bearish TK cross OR trend changes to downtrend
            if (close[i] < lower_cloud or not tk_cross_bullish or not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above cloud OR bullish TK cross OR trend changes to uptrend
            if (close[i] > upper_cloud or not tk_cross_bearish or not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals