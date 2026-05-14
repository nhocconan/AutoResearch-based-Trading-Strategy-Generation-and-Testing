#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context (1d)
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate Camarilla pivot levels (standard formula)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    daily_r1 = daily_close + daily_range * 1.1 / 12
    daily_s1 = daily_close - daily_range * 1.1 / 12
    daily_r2 = daily_close + daily_range * 1.1 / 6
    daily_s2 = daily_close - daily_range * 1.1 / 6
    daily_r3 = daily_close + daily_range * 1.1 / 4
    daily_s3 = daily_close - daily_range * 1.1 / 4
    daily_r4 = daily_close + daily_range * 1.1 / 2
    daily_s4 = daily_close - daily_range * 1.1 / 2
    
    # Align daily levels to 6h timeframe (wait for daily close)
    daily_pivot_6h = align_htf_to_ltf(prices, daily, daily_pivot)
    daily_r1_6h = align_htf_to_ltf(prices, daily, daily_r1)
    daily_s1_6h = align_htf_to_ltf(prices, daily, daily_s1)
    daily_r2_6h = align_htf_to_ltf(prices, daily, daily_r2)
    daily_s2_6h = align_htf_to_ltf(prices, daily, daily_s2)
    daily_r3_6h = align_htf_to_ltf(prices, daily, daily_r3)
    daily_s3_6h = align_htf_to_ltf(prices, daily, daily_s3)
    daily_r4_6h = align_htf_to_ltf(prices, daily, daily_r4)
    daily_s4_6h = align_htf_to_ltf(prices, daily, daily_s4)
    
    # Volume filter: current volume > 1.3x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # Volatility filter: use ATR(20) / close > 0.008
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / close
    volatility_filter = atr_ratio > 0.008
    
    signals = np.zeros(n)
    
    for i in range(300, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_pivot_6h[i]) or np.isnan(daily_r1_6h[i]) or 
            np.isnan(daily_s1_6h[i]) or np.isnan(daily_r2_6h[i]) or 
            np.isnan(daily_s2_6h[i]) or np.isnan(daily_r3_6h[i]) or 
            np.isnan(daily_s3_6h[i]) or np.isnan(daily_r4_6h[i]) or 
            np.isnan(daily_s4_6h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when both filters pass
        if volume_filter[i] and volatility_filter[i]:
            # Long: break above R3 with volume (continuation signal)
            if close[i] > daily_r3_6h[i]:
                signals[i] = 0.25
            # Short: break below S3 with volume (continuation signal)
            elif close[i] < daily_s3_6h[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_Volume_VolatilityFilter"
timeframe = "6h"
leverage = 1.0