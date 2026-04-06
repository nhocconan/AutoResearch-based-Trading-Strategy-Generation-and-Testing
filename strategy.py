#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with Daily Trend Filter.
# Uses Ichimoku Tenkan/Kijun cross for entry/exit on 6h timeframe.
# Daily EMA200 filter ensures trades align with higher timeframe trend.
# Volume confirmation (current volume > 1.5x 20-period average) filters weak signals.
# Works in both bull and bear markets via trend-following + mean reversion within trend.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_ichimoku_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily data
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2 + ema200_1d[i-1] * 199) / 201
    
    # Align daily EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = np.full(n, np.nan)
    for i in range(8, n):
        tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = np.full(n, np.nan)
    for i in range(25, n):
        kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = np.full(n, np.nan)
    for i in range(n):
        if not (np.isnan(tenkan[i]) or np.isnan(kijun[i])):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = np.full(n, np.nan)
    for i in range(51, n):
        senkou_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):
        # Skip if data not available
        if (np.isnan(ema200_aligned[i]) or np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Tenkan-Kijun cross down OR price below cloud OR stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            tenkan_kijun_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            below_cloud = close[i] < cloud_bottom
            
            if (tenkan_kijun_cross_down or below_cloud or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Tenkan-Kijun cross up OR price above cloud OR stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            tenkan_kijun_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            above_cloud = close[i] > cloud_top
            
            if (tenkan_kijun_cross_up or above_cloud or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Trend filter: price above/below daily EMA200
                price_above_ema = close[i] > ema200_aligned[i]
                price_below_ema = close[i] < ema200_aligned[i]
                
                # Tenkan-Kijun cross signals
                tenkan_kijun_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
                tenkan_kijun_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
                
                # Long: bullish cross above cloud in uptrend
                if (tenkan_kijun_cross_up and price_above_ema and 
                    close[i] > cloud_top):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bearish cross below cloud in downtrend
                elif (tenkan_kijun_cross_down and price_below_ema and 
                      close[i] < cloud_bottom):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals