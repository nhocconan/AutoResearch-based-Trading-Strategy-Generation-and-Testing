#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Donchian(20) breakout levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Calculate Donchian channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Align to daily timeframe (1d bars)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # === Daily volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_trend = ema_34_1w_aligned[i]
        upper_channel = high_20_aligned[i]
        lower_channel = low_20_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long on Donchian breakout + weekly uptrend + volume
            if (price_close > upper_channel and 
                price_close > weekly_trend and
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown + weekly downtrend + volume
            elif (price_close < lower_channel and 
                  price_close < weekly_trend and
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price retests the opposite Donchian level
            if position == 1 and price_close < lower_channel:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0