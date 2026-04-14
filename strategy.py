#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d Trend Filter (EMA 50)
# Williams %R identifies overbought/oversold conditions; mean reversion works in both bull/bear markets
# 1d EMA (50) filters trades to align with higher-timeframe trend, reducing false signals
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag
# Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) and price > 1d EMA50
# Short when %R > -20 (overbought) and price < 1d EMA50
# Exit when %R crosses above -50 (for long) or below -50 (for short)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams %R (14-period) on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for Williams %R calculation
    start = period
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: oversold (%R < -80) with uptrend filter
            if williams_r[i] < -80 and price > ema_1d_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: overbought (%R > -20) with downtrend filter
            elif williams_r[i] > -20 and price < ema_1d_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: %R crosses above -50 (momentum fading)
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: %R crosses below -50 (momentum fading)
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsR_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0