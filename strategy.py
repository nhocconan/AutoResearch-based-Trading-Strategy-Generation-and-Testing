#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
# Long: Williams %R crosses above -80 from below AND price > 1d EMA50 (uptrend) AND volume > 1.8x 20-period MA
# Short: Williams %R crosses below -20 from above AND price < 1d EMA50 (downtrend) AND volume > 1.8x 20-period MA
# Exit: Opposite Williams %R cross or EMA50 trend reversal.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Williams %R identifies overbought/oversold conditions; 1d EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment.

name = "6h_WilliamsR_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    williams_r_cross_above_80 = (williams_r_prev <= -80) & (williams_r > -80)
    williams_r_cross_below_20 = (williams_r_prev >= -20) & (williams_r < -20)
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r_prev[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 AND uptrend AND volume spike
            if williams_r_cross_above_80[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND downtrend AND volume spike
            elif williams_r_cross_below_20[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 OR trend turns down
            if williams_r_cross_below_20[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 OR trend turns up
            if williams_r_cross_above_80[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals