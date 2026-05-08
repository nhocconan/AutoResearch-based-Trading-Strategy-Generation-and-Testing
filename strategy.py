#\033[33m# WARNING: This is a mock response for illustration. Actual strategy must follow rules.\033[0m
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below AND 1d ADX > 25 AND volume > 1.3x 20-period average.
# Short when Williams %R crosses below -20 from above AND 1d ADX > 25 AND volume > 1.3x 20-period average.
# Exit when Williams %R crosses back to opposite extreme (-20 for long, -80 for short).
# This strategy captures mean-reversion in strong trends (ADX>25) to avoid chop losses.
# Williams %R identifies overbought/oversold levels. ADX ensures we only trade in strong trends.
# Volume filter confirms participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_1dADX25_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], 
                               np.abs(high_1d[1:] - close_1d[:-1])), 
                    np.abs(low_1d[1:] - close_1d[:-1]))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align ADX and Williams %R to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # ADX filter: strong trend (ADX > 25)
    adx_strong = adx_aligned > 25
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_strong[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below
            long_cross = (williams_r_aligned[i] > -80) and (williams_r_aligned[i-1] <= -80)
            # Short: Williams %R crosses below -20 from above
            short_cross = (williams_r_aligned[i] < -20) and (williams_r_aligned[i-1] >= -20)
            
            if long_cross and adx_strong[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif short_cross and adx_strong[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back above -20
            if (williams_r_aligned[i] > -20) and (williams_r_aligned[i-1] <= -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back below -80
            if (williams_r_aligned[i] < -80) and (williams_r_aligned[i-1] >= -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals