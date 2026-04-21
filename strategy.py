#!/usr/bin/env python3
"""
1d_1w_Ichimoku_Kijun_Cross_Volume_Momentum
Hypothesis: Weekly Ichimoku Kijun-sen acts as strong support/resistance. 
Cross above/below with daily volume confirmation captures trend changes. 
Works in bull/bear by using weekly trend filter and avoiding whipsaws with momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Ichimoku
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate Ichimoku Kijun-sen: (26-period high + low)/2
    # Using 26-week period for Kijun-sen
    high_26 = pd.Series(high_weekly).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_weekly).rolling(window=26, min_periods=26).min().values
    kijun_weekly = (high_26 + low_26) / 2.0
    
    # Align weekly Kijun to daily timeframe
    kijun_aligned = align_htf_to_ltf(prices, df_weekly, kijun_weekly)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily momentum: ROC(5) > 0 for momentum confirmation
    roc = np.zeros_like(close)
    roc[5:] = (close[5:] - close[:-5]) / close[:-5]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(kijun_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kijun = kijun_aligned[i]
        mom = roc[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        # Momentum filter: ROC > 0 for longs, ROC < 0 for shorts
        mom_long = mom > 0
        mom_short = mom < 0
        
        if position == 0:
            # Long: price crosses above Kijun with volume and momentum
            if price > kijun and vol_ok and mom_long:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Kijun with volume and momentum
            elif price < kijun and vol_ok and mom_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below Kijun
            if price < kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above Kijun
            if price > kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Ichimoku_Kijun_Cross_Volume_Momentum"
timeframe = "1d"
leverage = 1.0