#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d VWAP Trend Filter
# - Williams %R (14) on 4h for overbought/oversold signals
# - Long when Williams %R < -80 (oversold) and price > 1d VWAP (uptrend bias)
# - Short when Williams %R > -20 (overbought) and price < 1d VWAP (downtrend bias)
# - Williams %R captures short-term reversals; daily VWAP filters for institutional trend
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP on 1d timeframe (typical price * volume cumulative)
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # Align 1d VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Williams %R (14) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(vwap_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        wr = williams_r[i]
        vwap = vwap_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price > VWAP (uptrend)
            if wr < -80 and price > vwap:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price < VWAP (downtrend)
            elif wr > -20 and price < vwap:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or price falls below VWAP
            if wr > -50 or price < vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or price rises above VWAP
            if wr < -50 or price > vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dVWAP_TrendFilter"
timeframe = "4h"
leverage = 1.0