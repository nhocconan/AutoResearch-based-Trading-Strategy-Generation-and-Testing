#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian20_Breakout_WeekTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel and trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Weekly high, low, close for Donchian channel (20-period)
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate Donchian channel (20-period high/low)
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_w, donchian_low)
    
    # Weekly EMA(50) for trend filter
    close_w = pd.Series(weekly_close)
    ema50_w = close_w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume and above weekly EMA trend
            if close[i] > donchian_high_aligned[i] and vol_ok and close[i] > ema50_w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with volume and below weekly EMA trend
            elif close[i] < donchian_low_aligned[i] and vol_ok and close[i] < ema50_w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian low (reversion to mean)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian high (reversion to mean)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals