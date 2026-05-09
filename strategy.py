# 12h_PremiumDiscount_Equilibrium_1dTrend_Volume
# Hypothesis: Premium/discount to 1d VWAP with volume confirmation and 1d trend filter
# Works in bull/bear: mean reversion in range, trend continuation in trends
# Targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Uses 1d VWAP for mean reversion signals, 1d EMA for trend filter, volume for confirmation

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_PremiumDiscount_Equilibrium_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP, EMA trend, and volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d VWAP (volume-weighted average price)
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_numerator = (typical_price * df_1d['volume'].values).cumsum()
    vwap_denominator = df_1d['volume'].values.cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(vol_ma20_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Discount to VWAP (< -0.5%) with volume and above EMA trend
            if (close[i] < vwap_aligned[i] * 0.995) and vol_ok and (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Premium to VWAP (> +0.5%) with volume and below EMA trend
            elif (close[i] > vwap_aligned[i] * 1.005) and vol_ok and (close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Return to VWAP or trend reversal
            if (close[i] > vwap_aligned[i]) or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Return to VWAP or trend reversal
            if (close[i] < vwap_aligned[i]) or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals