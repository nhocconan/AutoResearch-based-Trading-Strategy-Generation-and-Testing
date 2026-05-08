#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_IchimokuCloud_Trend_DailyFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Senkou B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): 9-period high-low midpoint
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan_sen = ((max_high_tenkan + min_low_tenkan) / 2).values
    
    # Kijun-sen (Base Line): 26-period high-low midpoint
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun_sen = ((max_high_kijun + min_low_kijun) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): 52-period high-low midpoint, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2).values
    
    # Align Ichimoku components to 6h timeframe (wait for 1d close + 26-period forward shift)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish TK cross above cloud + volume confirmation
            bullish_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            above_cloud = close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]
            
            # Bearish TK cross below cloud + volume confirmation
            bearish_cross = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
            below_cloud = close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]
            
            if bullish_cross and above_cloud and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_cross and below_cloud and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross below cloud OR price drops below cloud
            exit_condition = (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) or \
                            (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross above cloud OR price rises above cloud
            exit_condition = (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) or \
                            (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals