#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1-week trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; in trending markets, 
# extreme readings can signal continuation rather than reversal. 
# Uses 1-week EMA for trend direction and volume surge for confirmation.
# Designed for fewer trades (target: 15-30/year) to minimize fee drag on 12h timeframe.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week trend filter: EMA(21) on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1-week high/low for Williams %R calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 14-period Williams %R on weekly data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    wsr = (highest_high - close_1w) / (highest_high - lowest_low) * -100
    wsr[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Align Williams %R to 12h timeframe (no extra delay needed as it's contemporaneous)
    wsr_aligned = align_htf_to_ltf(prices, df_1w, wsr)
    
    # Volume confirmation: current volume > 2.0x median of last 28 periods
    vol_median = pd.Series(volume).rolling(window=28, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(28, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(wsr_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Williams %R oversold (< -80) + uptrend (price > weekly EMA) + volume surge
        if (wsr_aligned[i] < -80 and close[i] > ema_1w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Williams %R overbought (> -20) + downtrend (price < weekly EMA) + volume surge
        elif (wsr_aligned[i] > -20 and close[i] < ema_1w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Williams %R returns to neutral range (-50 to -30 for longs, -70 to -50 for shorts)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and wsr_aligned[i] > -50) or
               (signals[i-1] == -0.25 and wsr_aligned[i] < -70))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsR_Trend_Volume"
timeframe = "12h"
leverage = 1.0