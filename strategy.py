#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Trend_1wFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1w data ONCE for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Ichimoku on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = df_1w['close'].values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1w, chikou)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 34, 4)  # Wait for Ichimoku, EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross above cloud in uptrend with volume
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            
            if tk_cross_up and price_above_cloud and uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below cloud in downtrend with volume
            elif tk_cross_down := (tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]):
                price_below_cloud = close[i] < cloud_bottom
                downtrend = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
                if price_below_cloud and downtrend and vol_condition:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: TK cross down or price falls into cloud
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_in_cloud = close[i] <= cloud_top and close[i] >= cloud_bottom
            
            if tk_cross_down or price_in_cloud or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross up or price rises into cloud
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_in_cloud = close[i] <= cloud_top and close[i] >= cloud_bottom
            
            if tk_cross_up or price_in_cloud or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with cloud filter on weekly timeframe, 
# combined with daily EMA trend and volume confirmation on 6h chart.
# - Weekly Ichimoku provides multi-timeframe trend and support/resistance
# - TK cross above/below cloud signals momentum shifts
# - Daily EMA(34) ensures alignment with longer-term trend
# - Volume spike confirms institutional participation
# - Works in bull markets (buy TK cross above cloud in uptrend) 
# - Works in bear markets (sell TK cross below cloud in downtrend)
# - Cloud acts as dynamic support/resistance reducing whipsaws
# - Position size 0.25 targets ~30-80 trades/year to avoid fee drag
# - Novel combination: Weekly Ichimoku + daily trend + volume (not recently tried)
# - Aims for 50-150 total trades over 4 years (12-37/year) within limits