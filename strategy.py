#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Donchian(10) breakout
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    donch_low = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Volume confirmation (12h)
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_12h['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
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
        vol_ratio_val = vol_ratio_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high + uptrend + volume
            if (price_close > upper_band and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.30
                position = 1
            # Short: breakdown below Donchian low + downtrend + volume
            elif (price_close < lower_band and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.30
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
                # Hold
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12h_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0