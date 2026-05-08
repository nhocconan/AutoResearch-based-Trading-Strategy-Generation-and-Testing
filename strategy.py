#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour weekly VWAP deviation with daily trend filter and volume spike
# Long when price crosses above weekly VWAP + 1*std, daily EMA(50) uptrend, volume spike
# Short when price crosses below weekly VWAP - 1*std, daily EMA(50) downtrend, volume spike
# Weekly VWAP provides institutional reference, daily EMA filters trend alignment
# Volume spike confirms breakout strength
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_WeeklyVWAP_1sigma_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly VWAP and standard deviation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_sum = (typical_price * df_1w['volume']).cumsum()
    volume_sum = df_1w['volume'].cumsum()
    vwap = vwap_sum / volume_sum
    # Calculate variance for standard deviation
    vwap_deviation = typical_price - vwap
    variance = (vwap_deviation ** 2 * df_1w['volume']).cumsum() / volume_sum
    vwap_std = np.sqrt(variance)
    
    # Upper and lower bands (VWAP ± 1*std)
    vwap_upper = vwap + vwap_std
    vwap_lower = vwap - vwap_std
    
    # Align weekly VWAP bands to 12h timeframe (available after weekly close)
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1w, vwap_upper.values)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1w, vwap_lower.values)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_upper_aligned[i]) or np.isnan(vwap_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vwap_upper_val = vwap_upper_aligned[i]
        vwap_lower_val = vwap_lower_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price crosses above VWAP + 1*std, daily uptrend, volume spike
            if price > vwap_upper_val and price > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below VWAP - 1*std, daily downtrend, volume spike
            elif price < vwap_lower_val and price < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below VWAP or daily trend turns down
            if price < vwap_upper_val or price < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above VWAP or daily trend turns up
            if price > vwap_lower_val or price > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals