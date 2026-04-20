#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WeeklyTrend_Filtered_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === Weekly Trend: EMA(34) for trend direction ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Donchian Channel (20) for breakout signals ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high/low with proper min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        trend_val = ema_34_1w_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(trend_val) or np.isnan(donchian_high_val) or 
            np.isnan(donchian_low_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with weekly uptrend and volume
            if close_val > donchian_high_val and trend_val > close_val and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with weekly downtrend and volume
            elif close_val < donchian_low_val and trend_val < close_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low OR weekly trend turns down
            if close_val < donchian_low_val or trend_val < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high OR weekly trend turns up
            if close_val > donchian_high_val or trend_val > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals