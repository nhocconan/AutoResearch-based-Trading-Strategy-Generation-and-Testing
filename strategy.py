#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchianBreakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly volume average (20-period)
    vol_1w = pd.Series(df_1w['volume'].values)
    vol_ma20_1w = vol_1w.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma20_1w)
    
    # Daily volume confirmation (20-period)
    vol_series = pd.Series(volume)
    vol_ma20_daily = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20_1w_aligned[i]) or 
            np.isnan(vol_ma20_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_daily[i]
        
        if position == 0:
            # Long: Break above weekly Donchian high with volume and above weekly EMA trend
            if (close[i] > donchian_high_aligned[i]) and vol_ok and (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low with volume and below weekly EMA trend
            elif (close[i] < donchian_low_aligned[i]) and vol_ok and (close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly EMA or returns to Donchian mid-point
            if (close[i] < ema50_1w_aligned[i]) or (close[i] < (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly EMA or returns to Donchian mid-point
            if (close[i] > ema50_1w_aligned[i]) or (close[i] > (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals