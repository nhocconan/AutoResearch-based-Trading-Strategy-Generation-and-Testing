#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA200 trend + volume spike
# Williams %R identifies overbought/oversold conditions for mean reversion
# 1d EMA200 ensures we trade with higher timeframe trend to avoid counter-trend whipsaws
# Volume spike (2.0x 20-period avg) confirms strong momentum behind moves
# Works in bull/bear: EMA200 trend filter avoids ranging market failures
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_williamsr_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend direction
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 14-period Williams %R
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if i < 13:
            williams_r[i] = np.nan
        else:
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high == lowest_low:
                williams_r[i] = -50.0  # Avoid division by zero
            else:
                williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1d EMA200 (trend change)
            if williams_r[i] > -20 or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 1d EMA200 (trend change)
            if williams_r[i] < -80 or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Williams %R extremes + EMA200 trend filter
            if volume_confirmed:
                # Long entry: Williams %R < -80 (oversold) AND price > 1d EMA200 (bullish pullback in uptrend)
                if williams_r[i] < -80 and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought) AND price < 1d EMA200 (bearish pullback in downtrend)
                elif williams_r[i] > -20 and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals