#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with daily trend filter
# - Fade at R3/S3 levels when price rejects extreme levels in ranging markets
# - Continuation breakout at R4/S4 levels when price breaks with daily trend alignment
# - Volume confirmation to avoid false breakouts
# - Works in both bull/bear: mean reversion in ranges, trend following in breaks
# Target: 50-150 total trades over 4 years (12-37/year)
# Size: 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Daily EMA trend filter (34-period)
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily levels to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need daily data and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R4_6h[i]) or np.isnan(R3_6h[i]) or 
            np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long conditions:
            # 1. Mean reversion: price rejects S3 level (long when price > S3 after being <= S3)
            # 2. OR breakout: price breaks above R4 with daily uptrend
            mean_reversion_long = (close[i-1] <= S3_6h[i-1] and close[i] > S3_6h[i])
            breakout_long = (close[i] > R4_6h[i] and ema_34_6h[i] > ema_34_6h[i-1])
            
            if (mean_reversion_long or breakout_long) and volume_filter:
                signals[i] = 0.25
                position = 1
            
            # Short conditions:
            # 1. Mean reversion: price rejects R3 level (short when price < R3 after being >= R3)
            # 2. OR breakout: price breaks below S4 with daily downtrend
            mean_reversion_short = (close[i-1] >= R3_6h[i-1] and close[i] < R3_6h[i])
            breakout_short = (close[i] < S4_6h[i] and ema_34_6h[i] < ema_34_6h[i-1])
            
            if (mean_reversion_short or breakout_short) and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches R3 (profit target) or S4 (stop)
            if close[i] >= R3_6h[i] or close[i] <= S4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S3 (profit target) or R4 (stop)
            if close[i] <= S3_6h[i] or close[i] >= R4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_R4_S4_MeanRev_Breakout"
timeframe = "6h"
leverage = 1.0