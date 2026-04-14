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
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily True Range for ATR
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # 14-day ATR for volatility measurement
    tr_series = pd.Series(tr)
    atr_14d = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: current volume > 1.3x 12-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=12, min_periods=12).mean().values
    
    # 6h Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate daily volatility as ATR normalized by price
    daily_volatility = atr_14d / close_1d
    daily_vol_series = pd.Series(daily_volatility)
    # Use 70th percentile of daily volatility over 14 days as threshold
    vol_threshold = daily_vol_series.rolling(window=14, min_periods=14).quantile(0.7).values
    # Align volatility threshold to 6h timeframe
    vol_threshold_6h = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_14d[i-1]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(vol_threshold_6h[i]):
            continue
        
        # Get previous day's data for volatility-based S1/R1 levels
        prev_close = close_1d[i-1]
        prev_atr = atr_14d[i-1]  # Previous day's ATR
        
        # Calculate volatility-adjusted threshold (0.3 * ATR) - tighter for fewer trades
        threshold = prev_atr * 0.3
        
        # Calculate dynamic S1/R1 levels based on volatility
        s1 = prev_close - threshold
        r1 = prev_close + threshold
        
        # Create arrays for alignment
        s1_array = np.full(len(df_1d), s1)
        r1_array = np.full(len(df_1d), r1)
        
        s1_6h = align_htf_to_ltf(prices, df_1d, s1_array)[i]
        r1_6h = align_htf_to_ltf(prices, df_1d, r1_array)[i]
        
        if position == 0:
            # Long: Price breaks above r1 with volume and high volatility regime
            if (close[i] > r1_6h and close[i-1] <= r1_6h and 
                volume[i] > vol_ma[i] * 1.3 and 
                daily_volatility[i] > vol_threshold_6h[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below s1 with volume and high volatility regime
            elif (close[i] < s1_6h and close[i-1] >= s1_6h and 
                  volume[i] > vol_ma[i] * 1.3 and 
                  daily_volatility[i] > vol_threshold_6h[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below s1 (reversal) or drops below Donchian low
            if close[i] < s1_6h or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above s1 (reversal) or rises above Donchian high
            if close[i] > s1_6h or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1d_VolatilityAdjusted_S1R1_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0