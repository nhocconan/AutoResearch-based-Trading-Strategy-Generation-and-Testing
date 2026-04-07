#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + Weekly Trend + Volume Spike
# Hypothesis: Williams %R identifies overbought/oversold conditions. In strong weekly trends,
# extreme readings (>80 for short, <20 for long) with volume confirmation capture mean reversion
# within the trend, working in both bull and bear markets by aligning with higher timeframe bias.
# Target: 60-120 total trades over 4 years (15-30/year).

name = "6h_williamsr_weekly_trend_volume_v1"
timeframe = "6h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold or trend turns bearish
            if williams_r[i] > -20 or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought or trend turns bullish
            if williams_r[i] < -80 or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Enter long: Williams %R oversold (< -80) in weekly uptrend
                if williams_r[i] < -80 and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: Williams %R overbought (> -20) in weekly downtrend
                elif williams_r[i] > -20 and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals