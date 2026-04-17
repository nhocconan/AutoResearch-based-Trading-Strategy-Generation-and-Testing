#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and ATR
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Ichimoku Cloud
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6x ATR for volatility filter (avoid extremely low volatility)
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 52  # Need enough data for Ichimoku calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above weekly EMA50 for bullish bias, below for bearish bias
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods (bottom 20% of ATR)
        vol_filter = atr_6h[i] > np.nanpercentile(atr_6h[max(0, i-50):i+1], 20) if i >= 50 else True
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + bullish weekly trend + volatility filter
            tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            
            if tk_cross_bullish and price_above_cloud and bullish_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + bearish weekly trend + volatility filter
            elif tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]:
                tk_cross_bearish = True
                price_below_cloud = close[i] < cloud_bottom
                if tk_cross_bearish and price_below_cloud and bearish_trend and vol_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud bottom
            tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_cross_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud top
            tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            
            if tk_cross_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_WeeklyTrend"
timeframe = "6h"
leverage = 1.0