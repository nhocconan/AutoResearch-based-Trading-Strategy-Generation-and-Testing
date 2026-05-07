#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_SMA50x200_BullBearMode_IchimokuCloud"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Ichimoku and 1w for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 52 or len(df_1w) < 52:
        return np.zeros(n)
    
    # 1d Ichimoku components
    high_9 = df_1d['high'].rolling(window=9).max().values
    low_9 = df_1d['low'].rolling(window=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = df_1d['high'].rolling(window=26).max().values
    low_26 = df_1d['low'].rolling(window=26).min().values
    kijun = (high_26 + low_26) / 2
    
    senkou_a = ((tenkan + kijun) / 2)
    high_52 = df_1d['high'].rolling(window=52).max().values
    low_52 = df_1d['low'].rolling(window=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Cloud: future senkou spans shifted forward 26 periods
    senkou_a_leading = np.roll(senkou_a, 26)
    senkou_b_leading = np.roll(senkou_b, 26)
    senkou_a_leading[:26] = np.nan
    senkou_b_leading[:26] = np.nan
    
    # Align Ichimoku to 6h
    tenkan_a = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_a = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_a = align_htf_to_ltf(prices, df_1d, senkou_a_leading)
    senkou_b_a = align_htf_to_ltf(prices, df_1d, senkou_b_leading)
    
    # 1w SMA50 and SMA200 for bull/bear mode
    sma50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma200_1w = pd.Series(df_1w['close']).rolling(window=200, min_periods=200).mean().values
    sma50_1w_a = align_htf_to_ltf(prices, df_1w, sma50_1w)
    sma200_1w_a = align_htf_to_ltf(prices, df_1w, sma200_1w)
    
    # Bull/bear mode: price above/below SMA200 on weekly
    bull_mode = sma50_1w_a > sma200_1w_a
    bear_mode = sma50_1w_a < sma200_1w_a
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_a[i]) or np.isnan(kijun_a[i]) or 
            np.isnan(senkou_a_a[i]) or np.isnan(senkou_b_a[i]) or
            np.isnan(sma50_1w_a[i]) or np.isnan(sma200_1w_a[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud color: green if senkou A > senkou B
        cloud_green = senkou_a_a[i] > senkou_b_a[i]
        cloud_red = senkou_a_a[i] < senkou_b_a[i]
        
        # TK cross: tenkan > kijun for bullish cross
        tk_bullish = tenkan_a[i] > kijun_a[i]
        tk_bearish = tenkan_a[i] < kijun_a[i]
        
        if position == 0:
            # Long: bull mode + TK bullish + price above cloud
            long_condition = bull_mode[i] and tk_bullish and cloud_green and (close[i] > senkou_a_a[i] and close[i] > senkou_b_a[i])
            # Short: bear mode + TK bearish + price below cloud
            short_condition = bear_mode[i] and tk_bearish and cloud_red and (close[i] < senkou_a_a[i] and close[i] < senkou_b_a[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK bearish or price enters cloud
            if tk_bearish or (close[i] < senkou_a_a[i] and close[i] < senkou_b_a[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK bullish or price enters cloud
            if tk_bullish or (close[i] > senkou_a_a[i] and close[i] > senkou_b_a[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals