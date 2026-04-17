#12345
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams %R (contrarian extreme)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R(14) on weekly: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 = overbought, < -80 = oversold
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    highest_high = high_1w_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_1w_series.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_1w_series) / (highest_high - lowest_low) * -100
    williams_r = williams_r.fillna(0).values  # Handle division by zero
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r, additional_delay_bars=0)
    
    # Get daily data for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R shows extreme oversold (< -80) and price above daily EMA50
            if williams_r_1w_aligned[i] < -80 and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R shows extreme overbought (> -20) and price below daily EMA50
            elif williams_r_1w_aligned[i] > -20 and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns from oversold (> -50) or price crosses below EMA50
            if williams_r_1w_aligned[i] > -50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns from overbought (< -50) or price crosses above EMA50
            if williams_r_1w_aligned[i] < -50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_Contrarian_EMA50"
timeframe = "6h"
leverage = 1.0