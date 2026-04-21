#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for 1d trend and structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Donchian(20) channels for breakout signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Volume confirmation: volume / 30-period average volume (12h)
    vol_ma_30 = pd.Series(df_12h['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio_12h = df_12h['volume'].values / vol_ma_30
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_band = donch_high_aligned[i]
        lower_band = donch_low_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.4  # Volume must be above average
        
        if position == 0:
            # Enter long: price breaks above Donchian high, uptrend, volume spike
            if (price_close > upper_band and 
                price_close > ema_trend and 
                vol_ratio > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, downtrend, volume spike
            elif (price_close < lower_band and 
                  price_close < ema_trend and 
                  vol_ratio > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout
            if position == 1 and price_close < lower_band:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0