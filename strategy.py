#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyTrend_6h_Pullback"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA Trend Filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower bounds
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        ema_50_val = ema_50_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_50_val) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1d EMA50 and breaking Donchian high with volume
            if close_val > ema_50_val and close_val > donch_high_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA50 and breaking Donchian low with volume
            elif close_val < ema_50_val and close_val < donch_low_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below Donchian low or trend reversal
            if close_val < donch_low_val or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above Donchian high or trend reversal
            if close_val > donch_high_val or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals