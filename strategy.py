#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Williams %R (14) identifies overbought/oversold conditions.
# Trend filter: 12h EMA34 (bullish if price > EMA, bearish if price < EMA).
# Volume filter: current volume > 1.5x 20-period average.
# Entries only during 08-20 UTC session to avoid low-liquidity hours.
# Target: 50-150 trades over 4 years (12-37/year) with disciplined entries.
# Williams %R logic: long when %R crosses above -80 from below, short when crosses below -20 from above.
# Exit when %R reverts to opposite threshold or trend filter fails.
# This captures mean reversion within the trend, avoiding choppy markets.

name = "6h_WilliamsR_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Williams %R (14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data for EMA34 trend (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND price above 12h EMA34 AND volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND price below 12h EMA34 AND volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (overbought) OR price breaks below 12h EMA34
            if (williams_r[i] < -20 and williams_r[i-1] >= -20) or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (oversold) OR price breaks above 12h EMA34
            if (williams_r[i] > -80 and williams_r[i-1] <= -80) or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals