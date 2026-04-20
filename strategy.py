#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI + 1d Williams %R Trend Filter
# - RSI(14) on 6h for momentum reversal signals (long <30, short >70)
# - Williams %R(14) on 1d for trend bias (long when > -50, short when < -50)
# - Combines short-term momentum reversal with intermediate trend filter
# - Designed to capture mean reversion in trending markets
# - Williams %R provides smoother trend signal than RSI
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d timeframe
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid div by zero
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate RSI(14) on 6h timeframe
    delta = pd.Series(prices['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi = rsi_14.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(rsi[i]) or np.isnan(williams_r_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].values[i]
        rsi_val = rsi[i]
        williams_r_val = williams_r_1d_aligned[i]
        
        if position == 0:
            # Long entry: RSI oversold (< 30) + Williams %R > -50 (bullish bias)
            if rsi_val < 30 and williams_r_val > -50:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (> 70) + Williams %R < -50 (bearish bias)
            elif rsi_val > 70 and williams_r_val < -50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI rises above 50 or Williams %R turns bearish
            if rsi_val > 50 or williams_r_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI falls below 50 or Williams %R turns bullish
            if rsi_val < 50 or williams_r_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_1dWilliamsR_TrendFilter"
timeframe = "6h"
leverage = 1.0