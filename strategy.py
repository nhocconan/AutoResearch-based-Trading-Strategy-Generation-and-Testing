#!/usr/bin/env python3
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
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA 34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift to avoid look-ahead
    donch_high = np.roll(donch_high, 1)
    donch_low = np.roll(donch_low, 1)
    donch_high[0] = np.nan
    donch_low[0] = np.nan
    
    # Daily volume ratio for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Exit conditions
        if position == 1:  # Long position
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # LONG: Break above Donchian upper with volume, in uptrend (weekly EMA)
            if price > upper and vol_ratio_val > 1.5 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Break below Donchian lower with volume, in downtrend
            elif price < lower and vol_ratio_val > 1.5 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0