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
    
    # Calculate daily ATR for volatility regime
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_series_1d = pd.Series(tr_1d)
    atr_1d = tr_series_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ATR percentile for regime filter (trending vs ranging)
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=50).quantile(0.5).values
    
    # Calculate 6h Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_1d[i]) or np.isnan(atr_percentile[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        # Get previous day's data for pivot calculation
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate pivot point and support/resistance levels
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_ = prev_high - prev_low
            
            # Support and resistance levels
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # Align pivot levels to 6h timeframe
            pivot_array = np.full(len(df_1d), pivot)
            s1_array = np.full(len(df_1d), s1)
            r1_array = np.full(len(df_1d), r1)
            s2_array = np.full(len(df_1d), s2)
            r2_array = np.full(len(df_1d), r2)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            
            pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_array)[i]
            s1_6h = align_htf_to_ltf(prices, df_1d, s1_array)[i]
            r1_6h = align_htf_to_ltf(prices, df_1d, r1_array)[i]
            s2_6h = align_htf_to_ltf(prices, df_1d, s2_array)[i]
            r2_6h = align_htf_to_ltf(prices, df_1d, r2_array)[i]
            s3_6h = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_6h = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            # Volume filter: current volume > 1.5x 10-period average
            vol_ma = np.mean(volume[max(0, i-10):i]) if i >= 10 else volume[i]
            
            # Regime filter: use daily ATR percentile to detect trending markets
            # Only trade in trending regimes (above median ATR)
            regime_filter = atr_1d[i] > atr_percentile[i]
            
            if position == 0:
                # Long: Price breaks above R2 with volume and in trending regime
                if (close[i] > r2_6h and close[i-1] <= r2_6h and 
                    volume[i] > vol_ma * 1.5 and 
                    close[i] > donchian_high[i] and  # Breakout confirmation
                    regime_filter):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below S2 with volume and in trending regime
                elif (close[i] < s2_6h and close[i-1] >= s2_6h and 
                      volume[i] > vol_ma * 1.5 and 
                      close[i] < donchian_low[i] and  # Breakdown confirmation
                      regime_filter):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below S1 (reversal) or drops below Donchian low
                if close[i] < s1_6h or close[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above R1 (reversal) or rises above Donchian high
                if close[i] > r1_6h or close[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_Pivot_R2_R1_Volume_Breakout_Regime"
timeframe = "6h"
leverage = 1.0