#!/usr/bin/env python3
name = "6h_1w_1d_Ichimoku_Cloud_Filter_Trend"
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
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily Ichimoku components
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(2)
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(2)
    
    # Align to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Weekly trend filter: EMA(21) on weekly close
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 4)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku cloud: green when span A > span B
        cloud_green = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        cloud_red = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, weekly uptrend, volume spike
            tk_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > senkou_span_a_aligned[i] and close[i] > senkou_span_b_aligned[i]
            weekly_uptrend = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            
            if tk_cross and price_above_cloud and cloud_green and weekly_uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, weekly downtrend, volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]) and \
                 (close[i] < senkou_span_a_aligned[i] and close[i] < senkou_span_b_aligned[i]) and \
                 (not cloud_green) and (ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1]) and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish or price drops below cloud
            tk_cross_bear = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
            price_below_cloud = close[i] < senkou_span_a_aligned[i] or close[i] < senkou_span_b_aligned[i]
            
            if tk_cross_bear or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish or price rises above cloud
            tk_cross_bull = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > senkou_span_a_aligned[i] and close[i] > senkou_span_b_aligned[i]
            
            if tk_cross_bull or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku TK cross with daily cloud filter and weekly trend
# - TK cross (Tenkan/Kijun crossover) provides timely entry signals
# - Only take trades in direction of the cloud (green=long, red=short)
# - Weekly EMA(21) filter ensures alignment with higher timeframe trend
# - Volume spike (2x average) confirms institutional participation
# - Works in bull markets (buy TK cross bullish in green cloud + weekly uptrend)
# - Works in bear markets (sell TK cross bearish in red cloud + weekly downtrend)
# - Exit on TK cross reversal or price exiting the cloud
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Ichimoku cloud acts as dynamic support/resistance with trend context
# - Weekly filter prevents counter-trend trades during major reversals