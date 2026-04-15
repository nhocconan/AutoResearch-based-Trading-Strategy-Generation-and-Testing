#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d/1w trend alignment
# Uses Tenkan/Kijun cross + Senkou Span cloud filter from daily timeframe
# Weekly trend filter (price vs weekly Kijun) to avoid counter-trend trades
# Designed for low frequency (15-30/year) with clear trend following in both bull/bear
# Ichimoku works well in crypto trends and avoids whipsaws in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max()
    tenkan_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max()
    kijun_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    senkou_high_52 = pd.Series(high_6h).rolling(window=52, min_periods=52).max()
    senkou_low_52 = pd.Series(low_6h).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((senkou_high_52 + senkou_low_52) / 2).shift(26)
    
    # Load 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Load 1w data for stronger trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w EMA25 for trend filter
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b.values)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema25_1w_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Long entry: TK cross bullish + price above cloud + bullish higher timeframe
        if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # Bullish TK cross
            close[i] > cloud_top and  # Price above cloud
            close[i] > ema50_1d_aligned[i] and  # Above daily EMA50
            close[i] > ema25_1w_aligned[i] and  # Above weekly EMA25
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: TK cross bearish + price below cloud + bearish higher timeframe
        elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # Bearish TK cross
              close[i] < cloud_bottom and  # Price below cloud
              close[i] < ema50_1d_aligned[i] and  # Below daily EMA50
              close[i] < ema25_1w_aligned[i] and  # Below weekly EMA25
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: TK cross reverses or price enters cloud
        elif position == 1 and (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                                close[i] < cloud_top):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                                 close[i] > cloud_bottom):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d1wTrend"
timeframe = "6h"
leverage = 1.0