#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Donchian channels (20-period) for breakout signals ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === 12h EMA50 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        upper_break = price_close > donch_high[i]
        lower_break = price_close < donch_low[i]
        trend_filter = ema_50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long on Donchian breakout + uptrend + volume
            if upper_break and price_close > trend_filter and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown + downtrend + volume
            elif lower_break and price_close < trend_filter and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite breakout or loss of trend
            if position == 1 and (lower_break or price_close < trend_filter):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (upper_break or price_close > trend_filter):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0