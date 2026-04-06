#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA(50) trend filter and volume confirmation
# Enter long when: Williams %R crosses above -20 (oversold reversal), price > 1d EMA(50), volume > 1.5x avg
# Enter short when: Williams %R crosses below -80 (overbought reversal), price < 1d EMA(50), volume > 1.5x avg
# Exit when: Williams %R returns to -50 (mean reversion) or opposite signal occurs
# Williams %R is effective in ranging markets (2025 bearish regime) while trend filter avoids counter-trend trades
# Target: 50-150 trades over 4 years with disciplined entries

name = "12h_williamsr_1dema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 12h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 14-period average
    volume_ma = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R returns to -50 (mean reversion) or bearish reversal
            if williams_r[i] >= -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R returns to -50 (mean reversion) or bullish reversal
            if williams_r[i] <= -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R reversal + trend filter + volume
            if volume[i] > volume_threshold[i]:
                # Bullish entry: Williams %R crosses above -20 from below
                if williams_r[i] > -20 and williams_r[i-1] <= -20 and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -80 from above
                elif williams_r[i] < -80 and williams_r[i-1] >= -80 and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals