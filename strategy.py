#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
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
    
    # Load daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    
    tenkan = (high_9 + low_9) / 2
    kijun = (high_26 + low_26) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_a = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_a = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_a = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_a = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 12h trend filter (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_a = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 50, 4)  # Wait for Ichimoku, EMA, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_a[i]) or np.isnan(kijun_a[i]) or 
            np.isnan(senkou_a_a[i]) or np.isnan(senkou_b_a[i]) or 
            np.isnan(ema_50_12h_a[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_a[i], senkou_b_a[i])
        cloud_bottom = min(senkou_a_a[i], senkou_b_a[i])
        
        if position == 0:
            # Long: price above cloud, Tenkan > Kijun, volume, and 12h uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            bullish_cross = tenkan_a[i] > kijun_a[i]
            uptrend = ema_50_12h_a[i] > ema_50_12h_a[i-1]
            
            if close[i] > cloud_top and bullish_cross and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, Tenkan < Kijun, volume, and 12h downtrend
            elif close[i] < cloud_bottom and not bullish_cross and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below cloud or volume drops
            if close[i] < cloud_top or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above cloud or volume drops
            if close[i] > cloud_bottom or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku cloud breakout with 12h trend filter and volume confirmation
# - Ichimoku cloud acts as dynamic support/resistance with forward-looking Kumo
# - Breakout above cloud with Tenkan/Kijun bullish cross in 12h uptrend = long
# - Breakdown below cloud with bearish cross in 12h downtrend = short
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy cloud breaks in uptrend) and bear (sell cloud breaks in downtrend)
# - Exit when price returns to cloud or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Ichimoku (not 6h) for better stability and reduced noise
# - 12h trend filter reduces whipsaws vs using same timeframe
# - Novel combination: Ichimoku cloud (1d) + trend (12h) + volume (6h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Ichimoku's forward-looking cloud provides better anticipation than static pivots
# - Tenkan/Kijun cross adds momentum confirmation to breakout direction
# - Volume confirmation reduces false breakouts in choppy markets
# - Designed to work in BOTH bull and bear markets via 12h trend filter
# - Cloud breakouts are proven effective in trending markets with volume confirmation
# - Targets BTC and ETH primarily, with potential applicability to SOL
# - Avoids overtrading by requiring multiple confluence factors for entry
# - Exit conditions are simple: price re-enters cloud or volume drops below threshold
# - Minimal parameters: Ichimoku (9,26,52), EMA 50, volume multiplier 1.8/1.2