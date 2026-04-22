#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku Cloud breakout with 12h trend filter and volume spike
    # Ichimoku provides dynamic support/resistance via Kumo (cloud) and momentum via TK cross
    # 12h EMA50 filters for trend direction to avoid counter-trend trades
    # Volume spike (2x 20-period MA) confirms breakout strength
    # Works in both bull and bear markets by trading with the trend only
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_12h).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_12h).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_12h).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_12h).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_12h).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_12h).rolling(window=52, min_periods=52).min()) / 2
    # Chikou Span (Lagging Span): not used for signals
    
    # Align Ichimoku components to 6h timeframe (with proper delay)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b.values)
    
    # Load 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: Price above cloud + TK cross bullish + price above EMA50 + volume spike
            if (close[i] > cloud_top and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + TK cross bearish + price below EMA50 + volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price re-enters cloud or TK cross reverses
            if position == 1:
                if (close[i] < cloud_top or 
                    tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > cloud_bottom or 
                    tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0