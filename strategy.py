#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Mean Reversion with Daily Trend Filter
# Uses Williams %R(14) for mean reversion entries (oversold/overbought) in the direction of the daily trend.
# In bull markets (price > daily SMA50): buy oversold (%R < -80), exit overbought (%R > -20).
# In bear markets (price < daily SMA50): sell overbought (%R > -20), exit oversold (%R < -80).
# Includes volume confirmation to filter low-probability signals. Target: 20-50 trades/year.
# Timeframe: 4h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate daily SMA50 for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align Williams %R and SMA50 to 4h timeframe (wait for daily close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(sma_50_aligned[i])):
            continue
            
        # Determine trend: bullish if price > SMA50, bearish if price < SMA50
        is_bullish = close_1d[-1] > sma_50[-1] if len(close_1d) > 0 else False  # Use latest daily values
        # More robust: use aligned values for current bar
        # We need to get the corresponding daily values for current 4h bar
        # Since we can't easily map, we'll use the trend from the most recent completed daily bar
        # For simplicity, we'll check if the current 4h close is above/below the aligned SMA50
        # This approximates the trend alignment
        
        # Long conditions: oversold (%R < -80) in bullish alignment + volume confirmation
        if (williams_r_aligned[i] < -80 and 
            close[i] > sma_50_aligned[i] and  # Price above daily SMA50 (bullish alignment)
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short conditions: overbought (%R > -20) in bearish alignment + volume confirmation
        elif (williams_r_aligned[i] > -20 and 
              close[i] < sma_50_aligned[i] and  # Price below daily SMA50 (bearish alignment)
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit conditions: reverse signal or extreme reversion
        elif position == 1 and (williams_r_aligned[i] > -20 or close[i] < sma_50_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r_aligned[i] < -80 or close[i] > sma_50_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_MeanReversion_DailyTrend"
timeframe = "4h"
leverage = 1.0