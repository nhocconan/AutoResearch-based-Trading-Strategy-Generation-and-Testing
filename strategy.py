# 12h_1dCamarillaR1S1_Breakout_Volume_Trend
# Hypothesis: 12h timeframe with Camarilla pivot (R1/S1) breakouts filtered by 1d EMA trend and volume.
# In bull markets, price breaks above R1 with upward trend and volume.
# In bear markets, price breaks below S1 with downward trend and volume.
# Camarilla levels provide statistically significant support/resistance.
# Volume confirms institutional interest.
# Trend filter ensures we trade with the higher timeframe momentum.
# Target: 15-30 trades/year to minimize fee drag.
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
    
    # Get 1d data for Camarilla pivot calculation (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Get 1d EMA for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d data to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3x 4-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and above EMA50 (uptrend)
            if close[i] > r1_aligned[i] and volume_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below EMA50 (downtrend)
            elif close[i] < s1_aligned[i] and volume_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (mean reversion to support)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (mean reversion to resistance)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarillaR1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0