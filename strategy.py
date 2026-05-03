#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h EMA50 trend filter and volume confirmation
# Long when %R < -80 (oversold), price > 12h EMA50, volume > 1.5x 20-bar average
# Short when %R > -20 (overbought), price < 12h EMA50, volume > 1.5x 20-bar average
# Uses Williams %R for momentum exhaustion, 12h EMA50 for trend alignment, volume for confirmation
# Designed for low trade frequency (~20-40/year on 4h) to minimize fee drag
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)

name = "4h_WilliamsR_Volume_12hEMA50_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R on 12h (period=14)
    highest_high = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume confirmation (1.5x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 14, 20) + 1  # EMA50(12h) + Williams%R(14) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold), price > 12h EMA50, volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought), price < 12h EMA50, volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (momentum fading) or price < 12h EMA50 (trend failure)
            if (williams_r_aligned[i] > -50 or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (momentum fading) or price > 12h EMA50 (trend failure)
            if (williams_r_aligned[i] < -50 or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals