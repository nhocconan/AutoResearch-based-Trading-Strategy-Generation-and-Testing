#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud with TK cross and volume confirmation
# - Uses Ichimoku components (Tenkan, Kijun, Senkou Span A/B) from daily timeframe
# - Long when Tenkan > Kijun (TK cross bullish) AND price above Cloud AND volume > 1.3x 20-period MA
# - Short when Tenkan < Kijun (TK cross bearish) AND price below Cloud AND volume > 1.3x 20-period MA
# - Cloud acts as dynamic support/resistance; TK cross provides momentum signal
# - Works in both bull/bear markets: trend-following with volatility filter via volume
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max() + 
                  pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max() + 
                 pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(2)  # Shifted 2 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = ((pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max() + 
                      pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()) / 2).shift(2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Volume filter: volume > 1.3x 20-period moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B form the cloud)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross conditions
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long entry: TK bullish cross AND price above cloud AND volume confirmation
            if tk_bullish and price_above_cloud and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: TK bearish cross AND price below cloud AND volume confirmation
            elif tk_bearish and price_below_cloud and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK bearish cross OR price drops below cloud bottom
            if not tk_bullish or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK bullish cross OR price rises above cloud top
            if not tk_bearish or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals