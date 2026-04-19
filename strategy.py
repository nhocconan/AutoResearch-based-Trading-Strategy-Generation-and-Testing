#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (price > 1d EMA50),
# we take counter-trend entries at extremes: long when %R < -80, short when %R > -20.
# In ranging markets, we fade the extremes: long at %R < -80, short at %R > -20.
# Volume confirmation: current volume > 1.5x 20-period average.
# Target: 20-30 trades/year per symbol to stay within frequency limits.
name = "12h_WilliamsR_EMATrend_Volume"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h data for Williams %R calculation
    # We need 14-period lookback, so we'll calculate on 12h data directly
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    highest_high_14 = highest_high(high, 14)
    lowest_low_14 = lowest_low(low, 14)
    
    # Calculate Williams %R
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100,
        -50  # Neutral when no range
    )
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # Ensure EMA50, Williams %R, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50[i//1]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema50_val = ema_50[i//1]  # 1d EMA value for current day
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend determination based on 1d EMA50
        is_uptrend = price > ema50_val
        is_downtrend = price < ema50_val
        
        if position == 0:
            # Look for entry signals
            if volume_confirmed:
                # Oversold condition: potential long
                if wr < -80:
                    signals[i] = 0.25
                    position = 1
                # Overbought condition: potential short
                elif wr > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral or overbought
            if wr > -50:  # Return to neutral or above
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral or oversold
            if wr < -50:  # Return to neutral or below
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals