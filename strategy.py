#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) and 1d EMA34 rising and volume > 1.3x average
# Short when Williams %R > -20 (overbought) and 1d EMA34 falling and volume > 1.3x average
# Williams %R identifies exhaustion points, EMA34 filters trend direction, volume confirms strength
# Designed for 6H timeframe to capture mean reversion in both bull and bear markets
# Targets 12-37 trades per year (50-150 over 4 years) to minimize fee drag

name = "6h_WilliamsR_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 and Williams %R ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        ema34 = ema34_1d_aligned[i]
        vol_ok = vol_conf[i]
        
        if position == 0:
            # Enter long: oversold, uptrend, volume confirmation
            if wr < -80 and ema34 > 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought, downtrend, volume confirmation
            elif wr > -20 and ema34 < 0 and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: overbought or trend turns down
            if wr > -20 or ema34 < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: oversold or trend turns up
            if wr < -80 or ema34 > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals