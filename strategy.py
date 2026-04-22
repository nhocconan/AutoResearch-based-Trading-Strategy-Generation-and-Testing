#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 200-day EMA trend filter with 1-week high/low breakout and volume confirmation.
# Works in bull/bear by using weekly trend direction (price above/below weekly 200-period EMA equivalent).
# Volume spike confirms breakout strength. Target: 10-25 trades/year (40-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend and breakout levels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 200-period EMA on weekly closes for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate weekly high and low for breakout levels (based on previous weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly high/low to daily timeframe (previous week's levels)
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(high_1w_aligned[i]) or 
            np.isnan(low_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly high in uptrend with volume
            if (close[i] > high_1w_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly low in downtrend with volume
            elif (close[i] < low_1w_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to weekly midpoint (average of weekly high and low)
            weekly_mid = (high_1w_aligned[i] + low_1w_aligned[i]) / 2.0
            if position == 1:
                if close[i] <= weekly_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= weekly_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyBreakout_200EMATrend_Volume"
timeframe = "1d"
leverage = 1.0