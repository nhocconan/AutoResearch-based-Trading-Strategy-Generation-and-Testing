#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume spike
# Uses 6h Williams %R(14) for mean reversion signals: long when %R < -80, short when %R > -20
# Filters trades by 1d EMA(50) trend: only long when price > EMA50, short when price < EMA50
# Requires volume > 1.5x 20-period 6h average for confirmation
# Exits when Williams %R returns to -50 (mean reversion completion)
# Position size 0.25 to limit drawdown
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) to minimize fee drag
# Works in both bull/bear: mean reversion in ranging markets, trend filter avoids counter-trend trades

name = "6h_1d_williamsr_trend_vol_v1"
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
    
    # Load 6h data ONCE before loop (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop (HTF filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 6h Williams %R (14-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    williams_r = np.full(len(df_6h), np.nan)
    for i in range(13, len(df_6h)):
        highest_high = np.max(high_6h[i-13:i+1])
        lowest_low = np.min(low_6h[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_6h[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Align 6h Williams %R to 6h timeframe (same timeframe, no shift needed)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])  # SMA seed
        for i in range(50, len(df_1d)):
            ema_50_1d[i] = (close_1d[i] * multiplier) + (ema_50_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA50 to 6h timeframe (use completed daily candles)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average on 6h (~5 days)
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion complete)
            if williams_r_aligned[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion complete)
            if williams_r_aligned[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            
            # Enter long: Williams %R oversold (< -80) + uptrend (price > EMA50) + volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: Williams %R overbought (> -20) + downtrend (price < EMA50) + volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_aligned[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals