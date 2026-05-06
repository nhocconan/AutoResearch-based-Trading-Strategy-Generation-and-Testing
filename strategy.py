#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Bollinger Band mean reversion with volume filter
# - Uses 4h Bollinger Bands (20, 2) for mean reversion signals
# - Uses 1h volume spike for entry confirmation
# - Uses session filter (08-20 UTC) to reduce noise
# - Enters long when price touches lower BB with volume confirmation
# - Enters short when price touches upper BB with volume confirmation
# - Exits when price returns to BB middle
# - Designed to capture mean reversion in ranging markets with institutional level respect
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing

name = "1h_4hBB_20_2_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Bollinger Bands (20, 2)
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # Align 4h Bollinger Bands to 1h timeframe
    upper_bb_1h = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_1h = align_htf_to_ltf(prices, df_4h, lower_bb)
    middle_bb_1h = align_htf_to_ltf(prices, df_4h, middle_bb)
    
    # Volume filter (1h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)  # Volume confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_bb_1h[i]) or np.isnan(lower_bb_1h[i]) or 
            np.isnan(middle_bb_1h[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB with volume confirmation
            if low[i] <= lower_bb_1h[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price touches upper BB with volume confirmation
            elif high[i] >= upper_bb_1h[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB
            if close[i] >= middle_bb_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to middle BB
            if close[i] <= middle_bb_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals