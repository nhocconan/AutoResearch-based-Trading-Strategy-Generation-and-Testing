#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation
# - Williams %R(14): measures overbought/oversold levels (-80 to -20 = oversold, -20 to 0 = overbought)
# - Long when %R crosses above -80 from below in 1w uptrend (close > EMA50) with volume spike
# - Short when %R crosses below -20 from above in 1w downtrend (close < EMA50) with volume spike
# - Uses 6h timeframe targeting 50-150 trades over 4 years (12-37/year) to minimize fee drag
# - 1w EMA50 filter ensures trading with higher timeframe trend direction
# - 6h volume > 1.5x 20-period average confirms breakout strength
# - Discrete position sizing (0.25) to minimize fee churn

name = "6h_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume confirmation: > 1.5x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (1.5 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 or trend/volume conditions fail
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 or trend/volume conditions fail
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R signals with trend and volume filters
            if vol_spike_1w_aligned[i]:
                # Long signal: %R crosses above -80 from below in 1w uptrend
                if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                    close_6h[i] > ema_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short signal: %R crosses below -20 from above in 1w downtrend
                elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                      close_6h[i] < ema_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals