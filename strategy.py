#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA50 Trend Filter + Volume Spike
# Enters long when Williams %R crosses above -80 from oversold, price above EMA50, volume > 2x average
# Enters short when Williams %R crosses below -20 from overbought, price below EMA50, volume > 2x average
# Exits when Williams %R returns to neutral zone (-50)
# Williams %R identifies momentum extremes; EMA50 filters for trend direction; volume confirms strength.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_WilliamsR_EMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R (14-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # Neutral when range is zero
    )
    
    # Volume confirmation (4h)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Get values
        close_val = close[i]
        williams_r_val = williams_r[i]
        ema_50_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Previous Williams %R for crossover detection
        prev_williams_r = williams_r[i-1]
        
        # Skip if any value is NaN
        if (np.isnan(williams_r_val) or np.isnan(ema_50_val) or 
            np.isnan(vol_ratio_val) or np.isnan(prev_williams_r)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 (from oversold), price above EMA50, volume spike
            if (williams_r_val > -80 and prev_williams_r <= -80 and 
                close_val > ema_50_val and vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 (from overbought), price below EMA50, volume spike
            elif (williams_r_val < -20 and prev_williams_r >= -20 and 
                  close_val < ema_50_val and vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (-50) or crosses below -50
            if williams_r_val <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (-50) or crosses above -50
            if williams_r_val >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals