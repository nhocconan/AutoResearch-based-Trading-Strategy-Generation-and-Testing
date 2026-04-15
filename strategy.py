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
    
    # Get daily data for pivot points and trend
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily pivot points (standard)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    daily_r2 = daily_pivot + (daily_high - daily_low)
    daily_s2 = daily_pivot - (daily_high - daily_low)
    
    # Align daily levels to 4h timeframe (wait for daily close)
    daily_pivot_4h = align_htf_to_ltf(prices, daily, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, daily, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, daily, daily_s1)
    daily_r2_4h = align_htf_to_ltf(prices, daily, daily_r2)
    daily_s2_4h = align_htf_to_ltf(prices, daily, daily_s2)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Volatility filter: avoid low volatility regimes
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio = atr / close
    volatility_filter = atr_ratio > 0.015
    
    # Trend filter: price above/below daily pivot
    trend_filter = close > daily_pivot_4h
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_pivot_4h[i]) or np.isnan(daily_r1_4h[i]) or 
            np.isnan(daily_s1_4h[i]) or np.isnan(daily_r2_4h[i]) or 
            np.isnan(daily_s2_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume and volatility filters pass
        if volume_filter[i] and volatility_filter[i]:
            # Long: price above R1 AND above daily pivot (trend)
            if close[i] > daily_r1_4h[i] and trend_filter[i]:
                signals[i] = 0.25
            # Short: price below S1 AND below daily pivot (trend)
            elif close[i] < daily_s1_4h[i] and not trend_filter[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyPivot_R1_S1_Trend_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0