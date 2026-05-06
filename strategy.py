#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R extremes with 4h Supertrend filter and volume confirmation
# Long when 12h Williams %R < -80 (oversold) AND 4h Supertrend is bullish AND volume > 2.0 * avg_volume(20)
# Short when 12h Williams %R > -20 (overbought) AND 4h Supertrend is bearish AND volume > 2.0 * avg_volume(20)
# Exit when 4h Supertrend flips direction (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R on 12h provides meaningful extreme readings less prone to whipsaw than lower TF
# 4h Supertrend ensures alignment with intermediate trend while adapting to volatility
# Volume confirmation filters weak reversals and ensures participation
# Works in bull (buying oversold dips in uptrend) and bear (selling overbought rallies in downtrend)

name = "4h_12hWilliamsR_Extreme_4hSupertrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for Williams %R calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = ((highest_high_12h - close_12h) / (highest_high_12h - lowest_low_12h)) * -100
    # Handle division by zero (when high == low)
    williams_r_12h = np.where((highest_high_12h - lowest_low_12h) == 0, -50, williams_r_12h)
    
    # Get 4h data ONCE before loop for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:  # Need sufficient data for ATR calculation
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend (10-period ATR, multiplier 3.0)
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend upper and lower bands
    hl2_4h = (high_4h + low_4h) / 2.0
    upper_band_4h = hl2_4h + (3.0 * atr_4h)
    lower_band_4h = hl2_4h - (3.0 * atr_4h)
    
    # Initialize Supertrend
    supertrend_4h = np.full_like(close_4h, np.nan)
    uptrend_4h = np.full_like(close_4h, True)  # Start assuming uptrend
    
    # Calculate Supertrend iteratively (need previous values)
    for i in range(1, len(close_4h)):
        if np.isnan(atr_4h[i]) or np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i]):
            supertrend_4h[i] = np.nan
            uptrend_4h[i] = uptrend_4h[i-1]
            continue
            
        # Upper band logic
        if upper_band_4h[i] < upper_band_4h[i-1] or close_4h[i-1] > upper_band_4h[i-1]:
            upper_band_4h[i] = upper_band_4h[i]
        else:
            upper_band_4h[i] = upper_band_4h[i-1]
            
        # Lower band logic
        if lower_band_4h[i] > lower_band_4h[i-1] or close_4h[i-1] < lower_band_4h[i-1]:
            lower_band_4h[i] = lower_band_4h[i]
        else:
            lower_band_4h[i] = lower_band_4h[i-1]
            
        # Trend logic
        if uptrend_4h[i-1]:
            supertrend_4h[i] = lower_band_4h[i]
            if close_4h[i] < lower_band_4h[i]:
                uptrend_4h[i] = False
            else:
                uptrend_4h[i] = True
        else:
            supertrend_4h[i] = upper_band_4h[i]
            if close_4h[i] > upper_band_4h[i]:
                uptrend_4h[i] = True
            else:
                uptrend_4h[i] = False
    
    # Align 12h Williams %R to 4h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    
    # Align 4h Supertrend to 4h timeframe (wait for completed 4h bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    uptrend_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND Supertrend bullish AND volume confirmation
            if (williams_r_aligned[i] < -80 and uptrend_aligned[i] > 0.5 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND Supertrend bearish AND volume confirmation
            elif (williams_r_aligned[i] > -20 and uptrend_aligned[i] < 0.5 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Supertrend flips bearish
            if uptrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Supertrend flips bullish
            if uptrend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals