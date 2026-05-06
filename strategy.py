#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d EMA34 trend filter and volume spike
# Long when Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume > 2.0 * 20-period avg volume
# Short when Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume > 2.0 * 20-period avg volume
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R captures momentum extremes, EMA34 filters trend direction, volume confirms participation

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Williams %R calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with uptrend and volume spike
            if (williams_r_aligned[i] < -80.0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with downtrend and volume spike
            elif (williams_r_aligned[i] > -20.0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50
            if williams_r_aligned[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50
            if williams_r_aligned[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals