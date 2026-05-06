#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (R3/S3) with 1d trend filter and volume confirmation
# - Uses weekly Camarilla pivot levels (R3/S3) from 1d data for mean-reversion entries
# - Uses 1d EMA34 to determine trend direction (long above EMA34, short below)
# - Uses 6h volume spike (>2x 20-period average) for entry confirmation
# - Enters long when price touches weekly S3 in uptrend (price > 1d EMA34) with volume
# - Enters short when price touches weekly R3 in downtrend (price < 1d EMA34) with volume
# - Exits when price reaches weekly pivot (P) or opposite extreme (R3/S3)
# - Designed to capture mean-reversion in trending markets with institutional level respect
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyCamarilla_R3S3_1dTrend_Volume"
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
    
    # Get 1d data for weekly pivot calculation (need 5 days for weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot points from daily data
    # Group daily data into weeks (Monday to Friday)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high, low, close (using 5-day periods)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Calculate pivot point (P)
    P = (weekly_high + weekly_low + weekly_close) / 3
    
    # Calculate Camarilla levels
    # R3 = P + 1.1 * (weekly_high - weekly_low)
    # S3 = P - 1.1 * (weekly_high - weekly_low)
    R3 = P + 1.1 * (weekly_high - weekly_low)
    S3 = P - 1.1 * (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    P_6h = align_htf_to_ltf(prices, df_1d, P)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(P_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S3 in uptrend (price > EMA34) with volume
            if close[i] <= S3_6h[i] and close[i] > ema_34_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches R3 in downtrend (price < EMA34) with volume
            elif close[i] >= R3_6h[i] and close[i] < ema_34_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches pivot (P) or touches R3
            if close[i] >= P_6h[i] or close[i] >= R3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches pivot (P) or touches S3
            if close[i] <= P_6h[i] or close[i] <= S3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals