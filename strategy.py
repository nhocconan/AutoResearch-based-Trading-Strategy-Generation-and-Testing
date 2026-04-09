#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R + 4h EMA trend + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 4h EMA (50) provides trend filter: long only when price > EMA50, short when price < EMA50
# Volume confirmation ensures momentum behind moves
# Works in bull/bear: Williams %R captures reversals, EMA filter avoids counter-trend trades
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.20

name = "1h_4h_williamsr_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA for 4h trend
    close_4h = df_4h['close'].values
    ema_50 = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_50[i] = (close_4h[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 4h EMA to 1h timeframe (wait for 4h bar close)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Williams %R (14-period) on 1h data
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if i < 13:
            williams_r[i] = np.nan
        else:
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                williams_r[i] = -50  # avoid division by zero
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 4h EMA50 (trend change)
            if williams_r[i] > -20 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 4h EMA50 (trend change)
            if williams_r[i] < -80 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume confirmation and Williams %R + 4h EMA filter
            if volume_confirmed:
                # Long entry: Williams %R < -80 (oversold) AND price > 4h EMA50 (uptrend)
                if williams_r[i] < -80 and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: Williams %R > -20 (overbought) AND price < 4h EMA50 (downtrend)
                elif williams_r[i] > -20 and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals