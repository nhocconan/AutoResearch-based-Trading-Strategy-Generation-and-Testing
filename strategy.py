#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Uses Ichimoku TK cross and cloud for momentum, filtered by 1d EMA(50) trend.
# Volume > 1.5x average confirms institutional participation.
# Works in bull/bear as 1d EMA adapts to trend.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku components (6h)
    conv_period = 9   # Tenkan-sen
    base_period = 26  # Kijun-sen
    span_b_period = 52 # Senkou span B
    
    # Tenkan-sen: (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=conv_period, min_periods=conv_period).max() + 
                  pd.Series(low).rolling(window=conv_period, min_periods=conv_period).min()) / 2
    
    # Kijun-sen: (highest high + lowest low)/2 for past 26 periods
    kijun_sen = (pd.Series(high).rolling(window=base_period, min_periods=base_period).max() + 
                 pd.Series(low).rolling(window=base_period, min_periods=base_period).min()) / 2
    
    # Senkou span A: (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(base_period)
    
    # Senkou span B: (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=span_b_period, min_periods=span_b_period).max() + 
                      pd.Series(low).rolling(window=span_b_period, min_periods=span_b_period).min()) / 2).shift(base_period)
    
    # Current price relative to cloud
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_span_a.values, senkou_span_b.values)
    cloud_bottom = np.minimum(senkou_span_a.values, senkou_span_b.values)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=26, min_periods=26).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(base_period, span_b_period, 26)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen.iloc[i]) or 
            np.isnan(kijun_sen.iloc[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Ichimoku signals
        tk_cross = tenkan_sen.iloc[i] > kijun_sen.iloc[i]  # Bullish TK cross
        price_above_cloud = close[i] > cloud_top[i]        # Price above cloud
        price_below_cloud = close[i] < cloud_bottom[i]     # Price below cloud
        
        if position == 0:
            # Enter long: Bullish TK cross + price above cloud + above 1d EMA + volume
            if (tk_cross and 
                price_above_cloud and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Bearish TK cross + price below cloud + below 1d EMA + volume
            elif (not tk_cross and 
                  price_below_cloud and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bearish TK cross or price drops below cloud
            if (not tk_cross or close[i] < cloud_top[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bullish TK cross or price rises above cloud
            if (tk_cross or close[i] > cloud_bottom[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_EMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0