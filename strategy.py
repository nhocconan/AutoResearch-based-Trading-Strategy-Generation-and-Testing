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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h
    upper_20_1h = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_1h = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # Get 1d HTF data for daily pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points from prior day (using 1d data)
    daily_high = df_1d['high'].shift(1).values
    daily_low = df_1d['low'].shift(1).values
    daily_close = df_1d['close'].shift(1).values
    
    # Daily pivot: (H+L+C)/3
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    # Daily R1: 2*P - L
    daily_r1 = 2 * daily_pivot - daily_low
    # Daily S1: 2*P - H
    daily_s1 = 2 * daily_pivot - daily_high
    
    # Align daily pivot levels to 1h
    daily_pivot_1h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_1h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_1h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate 1h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1h[i]) or np.isnan(lower_20_1h[i]) or 
            np.isnan(daily_pivot_1h[i]) or np.isnan(daily_r1_1h[i]) or 
            np.isnan(daily_s1_1h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. Price above daily pivot (bullish bias from prior day)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_1h[i] and
            close[i] > daily_pivot_1h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 1h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. Price below daily pivot (bearish bias from prior day)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_1h[i] and
              close[i] < daily_pivot_1h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_Donchian20_1d_DailyPivot_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0