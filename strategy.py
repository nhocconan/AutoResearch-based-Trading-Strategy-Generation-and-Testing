#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Uses 12h primary timeframe for lower trade frequency (target: 50-150 total trades over 4 years)
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels work in both bull/bear regimes
# 1d EMA50 trend filter ensures trades align with higher timeframe momentum
# Volume confirmation (1.8x 20-period average) reduces false signals
# Designed with tight entry conditions to minimize fee drag while maintaining edge
# Target: 60-100 total trades over 4 years (15-25/year) - within proven winning range for 12h

name = "12h_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams %R (14-period) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, EMA, and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from oversold + price > 1d EMA50 + volume confirm
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought + price < 1d EMA50 + volume confirm
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) or crosses above -20 (overbought)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            elif williams_r[i] > -20 and williams_r[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) or crosses below -80 (oversold)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            elif williams_r[i] < -80 and williams_r[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals