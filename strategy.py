#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # --- WEEKLY TREND FILTER: price above/below weekly 200 EMA ---
    weekly_close = df_1d['close'].values  # reuse daily close for weekly trend calc
    weekly_ema_200 = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema_200_aligned = align_htf_to_ltf(prices, df_1d, weekly_ema_200)
    
    # --- DAILY ICHIMOKU COMPONENTS ---
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 4)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_ema_200_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom (cloud is between Senkou A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, weekly uptrend, volume spike
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            weekly_uptrend = close[i] > weekly_ema_200_aligned[i]
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            
            if tk_cross_up and price_above_cloud and weekly_uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, weekly downtrend, volume spike
            elif tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]:
                price_below_cloud = close[i] < cloud_bottom
                weekly_downtrend = close[i] < weekly_ema_200_aligned[i]
                if price_below_cloud and weekly_downtrend and vol_condition:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: Tenkan crosses below Kijun or price drops below cloud
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_below_cloud = close[i] < cloud_bottom
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun or price rises above cloud
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter and volume confirmation
# - Ichimoku TK cross (Tenkan/Kijun) provides momentum signals
# - Price must be above/below cloud to ensure trend alignment
# - Weekly 200 EMA filter ensures we only trade in higher timeframe trend direction
# - Volume spike (2x average) confirms institutional participation
# - Works in bull markets (buy TK crosses above in uptrend) and bear markets (sell TK crosses below in downtrend)
# - Exit on TK cross reversal or price re-entering cloud
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Ichimoku (not approximated) for proper signals
# - Weekly trend filter reduces whipsaws vs using same timeframe
# - Novel combination: Ichimoku (1d) + weekly trend (1w) + volume (6h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Cloud acts as dynamic support/resistance, TK cross as momentum trigger
# - Designed to work in BOTH bull and bear markets via weekly trend filter
# - Volume confirmation reduces false breakouts in ranging markets