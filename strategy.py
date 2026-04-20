#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA Trend and 1d Volume Filter
# Uses Williams %R (14) for mean-reversion entries on 1h timeframe
# Enters long when Williams %R < -80 and price > 4h EMA50 (uptrend)
# Enters short when Williams %R > -20 and price < 4h EMA50 (downtrend)
# Requires 1d volume > 1.5x 20-day average for institutional confirmation
# Exits when Williams %R returns to -50 (mean reversion completion)
# Williams %R identifies overbought/oversold conditions; EMA50 filters trend direction;
# Volume filter ensures institutional participation. Target: 15-30 trades/year.

name = "1h_WilliamsR_4hEMA50_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 1d volume average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Williams %R on 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when HH == LL
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_50_val = ema_50_aligned[i]
        williams_r_val = williams_r[i]
        vol_ma_20_val = vol_ma_20_aligned[i]
        volume_val = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_50_val) or np.isnan(williams_r_val) or 
            np.isnan(vol_ma_20_val) or np.isnan(volume_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1h volume should be significant relative to daily average
        # Approximate: 1h volume > (daily average / 24) * 1.5
        vol_threshold = vol_ma_20_val / 24.0 * 1.5
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80), price above EMA50, volume sufficient
            if williams_r_val < -80 and close_val > ema_50_val and volume_val > vol_threshold:
                signals[i] = 0.20
                position = 1
            # Short entry: Williams %R overbought (> -20), price below EMA50, volume sufficient
            elif williams_r_val > -20 and close_val < ema_50_val and volume_val > vol_threshold:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion)
            if williams_r_val >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion)
            if williams_r_val <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals