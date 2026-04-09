#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion
# 1d EMA filter ensures we only take trades in direction of higher timeframe trend
# Volume confirmation avoids low-liquidity false signals
# Works in bull/bear: EMA filter adapts to trend, Williams %R captures reversals within trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        (highest_high - close) / (highest_high - lowest_low) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x 1d average volume
        volume_confirmed = volume[i] > 1.2 * avg_volume_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR trend turns bearish
            if williams_r[i] > -20 or close[i] < ema_34_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR trend turns bullish
            if williams_r[i] < -80 or close[i] > ema_34_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: only trade in direction of 1d EMA trend
            if close[i] > ema_34_1d_aligned[i]:  # Bullish trend
                # Look for oversold conditions to go long
                if williams_r[i] < -80 and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
            else:  # Bearish trend
                # Look for overbought conditions to go short
                if williams_r[i] > -20 and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals