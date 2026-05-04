#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume confirmation
# Long when Williams %R crosses above -80 from oversold, price > 1d EMA34, and volume > 1.5x 20-period volume EMA
# Short when Williams %R crosses below -20 from overbought, price < 1d EMA34, and volume > 1.5x 20-period volume EMA
# Williams %R identifies short-term reversals; 1d EMA34 filters for intermediate-term trend alignment to reduce whipsaw
# Volume confirmation ensures breakout/retest validity. Targets 12-37 trades/year on 6h with strict entry conditions.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "6h_WilliamsR_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for HTF EMA34 and Williams %R calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R on 1d data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Align Williams %R to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (from oversold) AND price > 1d EMA34 AND volume spike
            williams_r_prev = williams_r_1d_aligned[i-1] if i > 0 else -100
            williams_r_cross_up = (williams_r_prev <= -80) and (williams_r_1d_aligned[i] > -80)
            if williams_r_cross_up and (close[i] > ema34_1d_aligned[i]) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (from overbought) AND price < 1d EMA34 AND volume spike
            elif (williams_r_prev >= -20) and (williams_r_1d_aligned[i] < -20) and (close[i] < ema34_1d_aligned[i]) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) OR price < 1d EMA34 (trend break)
            williams_r_prev = williams_r_1d_aligned[i-1] if i > 0 else -100
            williams_r_cross_down = (williams_r_prev >= -50) and (williams_r_1d_aligned[i] < -50)
            if williams_r_cross_down or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) OR price > 1d EMA34 (trend break)
            williams_r_prev = williams_r_1d_aligned[i-1] if i > 0 else -100
            williams_r_cross_up = (williams_r_prev <= -50) and (williams_r_1d_aligned[i] > -50)
            if williams_r_cross_up or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals