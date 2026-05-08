#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold reversal) AND price > 12h EMA50 AND 4h volume > 1.5x 20-period average.
# Short when Williams %R crosses below -80 (overbought reversal) AND price < 12h EMA50 AND 4h volume > 1.5x 20-period average.
# Exit when Williams %R crosses back below -80 (long) or above -20 (short).
# Williams %R is effective in ranging markets (2025+), with trend filter to avoid counter-trend trades.
# Volume confirmation ensures momentum behind reversals.
# Target: 80-150 total trades over 4 years (20-37/year) with controlled frequency.

name = "4h_WilliamsR_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20, above 12h EMA50, volume spike
            williams_cross_up = (williams_r[i] > -20) and (williams_r[i-1] <= -20)
            long_cond = williams_cross_up and (close[i] > ema50_12h_aligned[i]) and volume_filter[i]
            # Short conditions: Williams %R crosses below -80, below 12h EMA50, volume spike
            williams_cross_down = (williams_r[i] < -80) and (williams_r[i-1] >= -80)
            short_cond = williams_cross_down and (close[i] < ema50_12h_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back below -80 (overbought)
            if williams_r[i] < -80 and williams_r[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back above -20 (oversold)
            if williams_r[i] > -20 and williams_r[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals