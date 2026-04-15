#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate daily Ichimoku components (based on previous day)
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over 9 periods
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over 26 periods
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over 52 periods
    
    # Tenkan-sen (9-period)
    tenkan_sen_9 = pd.Series(daily_high).rolling(window=9, min_periods=9).max().values
    tenkan_sen_1 = pd.Series(daily_low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (tenkan_sen_9 + tenkan_sen_1) / 2
    
    # Kijun-sen (26-period)
    kijun_sen_26 = pd.Series(daily_high).rolling(window=26, min_periods=26).max().values
    kijun_sen_2 = pd.Series(daily_low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (kijun_sen_26 + kijun_sen_2) / 2
    
    # Senkou Span A
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (52-period)
    senkou_span_b_52 = pd.Series(daily_high).rolling(window=52, min_periods=52).max().values
    senkou_span_b_2 = pd.Series(daily_low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (senkou_span_b_52 + senkou_span_b_2) / 2
    
    # Align HTF Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Cloud top = max(Senkou Span A, Senkou Span B)
        # Cloud bottom = min(Senkou Span A, Senkou Span B)
        cloud_top = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Entry conditions:
        # 1. TK Cross: Tenkan-sen crosses above/below Kijun-sen
        # 2. Price breaks above/below cloud with volume confirmation
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: Bullish TK cross + price above cloud
        if (tenkan_sen_6h[i] > kijun_sen_6h[i] and          # Bullish TK cross
            close[i] > cloud_top and                         # Price above cloud
            volume_ratio[i] > 1.3 and                        # Volume confirmation
            atr_14[i] > 0.005 * close[i]):                   # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: Bearish TK cross + price below cloud
        elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and       # Bearish TK cross
              close[i] < cloud_bottom and                    # Price below cloud
              volume_ratio[i] > 1.3 and                      # Volume confirmation
              atr_14[i] > 0.005 * close[i]):                 # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0