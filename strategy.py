#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Williams %R extreme levels with 1d EMA50 trend filter and volume spike confirmation
# Long when Williams %R crosses above -20 (from below) AND price > 1d EMA50 AND volume > 1.8 * avg_volume(24) on 6h
# Short when Williams %R crosses below -80 (from above) AND price < 1d EMA50 AND volume > 1.8 * avg_volume(24) on 6h
# Exit when Williams %R crosses opposite extreme (-80 for long, -20 for short) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets
# 1d EMA50 filters primary trend to avoid counter-trend trades during strong trends
# Volume spike confirms reversal strength and reduces false signals
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)

name = "6h_WilliamsR_Extreme_Reversal_1dEMA50_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume confirmation: volume > 1.8 * 24-period average volume on 6h
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * avg_volume_24)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(avg_volume_24[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 (from below), price > 1d EMA50, volume confirmation, in session
            if williams_r[i] > -20 and williams_r[i-1] <= -20 and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 (from above), price < 1d EMA50, volume confirmation, in session
            elif williams_r[i] < -80 and williams_r[i-1] >= -80 and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -80 OR volume drops below average
            if williams_r[i] < -80 and williams_r[i-1] >= -80 or volume[i] < avg_volume_24[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -20 OR volume drops below average
            if williams_r[i] > -20 and williams_r[i-1] <= -20 or volume[i] < avg_volume_24[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals