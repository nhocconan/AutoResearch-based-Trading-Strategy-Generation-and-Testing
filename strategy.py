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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR (14-period) for volatility filter
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_series_1d = pd.Series(tr_1d)
    atr_1d = tr_series_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile (60th) over 30 days for volatility filter
    atr_series_1d = pd.Series(atr_1d)
    atr_percentile = atr_series_1d.rolling(window=30, min_periods=30).quantile(0.6).values
    volatility_filter = atr_1d > atr_percentile
    
    # Calculate 4h volume filter: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_1d[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(volatility_filter[i]):
            continue
        
        # Get previous day's data for pivot calculation
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate pivot point and support/resistance levels
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_ = prev_high - prev_low
            
            # Focus on S1/R1 for tighter entries (fewer trades)
            s1 = pivot - range_
            r1 = pivot + range_
            
            # Align S1/R1 levels to 12h timeframe
            s1_array = np.full(len(df_1d), s1)
            r1_array = np.full(len(df_1d), r1)
            
            s1_12h = align_htf_to_ltf(prices, df_1d, s1_array)[i]
            r1_12h = align_htf_to_ltf(prices, df_1d, r1_array)[i]
            
            if position == 0:
                # Long: Price breaks above R1 with volume and in volatile regime
                if (close[i] > r1_12h and close[i-1] <= r1_12h and 
                    volume[i] > vol_ma[i] * 1.3 and 
                    volatility_filter[i]):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below S1 with volume and in volatile regime
                elif (close[i] < s1_12h and close[i-1] >= s1_12h and 
                      volume[i] > vol_ma[i] * 1.3 and 
                      volatility_filter[i]):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below S1 (reversal) or drops below Donchian low
                if close[i] < s1_12h or close[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above S1 (reversal) or rises above Donchian high
                if close[i] > s1_12h or close[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "4h_1d_S1R1_Breakout_Vol_VolatilityFilter_v4"
timeframe = "4h"
leverage = 1.0