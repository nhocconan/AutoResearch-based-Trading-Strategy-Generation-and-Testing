#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below in bull trend (close > 12h EMA50) with volume > 1.8x 20-period MA.
# Short when Williams %R crosses below -20 from above in bear trend (close < 12h EMA50) with volume spike.
# Williams %R identifies overextended conditions; 12h EMA50 filters whipsaw in ranging markets; volume confirms participation.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
# Works in both bull and bear by following the 12h trend direction.

name = "6h_WilliamsR_12hEMA50_Volume"
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
    
    # Get 12h data for Williams %R and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_12h['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        wr = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 from below in bull trend with volume spike
            if is_bull_trend and wr > -80 and vol_spike:
                # Check for crossover: previous Williams %R <= -80
                if i > 100 and not np.isnan(williams_r_aligned[i-1]) and williams_r_aligned[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R crosses below -20 from above in bear trend with volume spike
            elif is_bear_trend and wr < -20 and vol_spike:
                # Check for crossover: previous Williams %R >= -20
                if i > 100 and not np.isnan(williams_r_aligned[i-1]) and williams_r_aligned[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR trend reversal
            if wr < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR trend reversal
            if wr > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals