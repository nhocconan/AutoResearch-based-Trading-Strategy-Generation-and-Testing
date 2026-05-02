#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Elder Ray (Bull/Bear Power) trend filter + volume confirmation
# Uses 6h primary timeframe for Williams %R oversold/overbought signals (mean reversion)
# 1d Elder Ray confirms trend direction: Bull Power > 0 for longs, Bear Power < 0 for shorts
# Volume confirmation (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R captures short-term exhaustion, Elder Ray filters by higher-timeframe trend, volume confirms conviction
# Works in both bull and bear markets by only trading mean reversion in direction of 1d trend

name = "6h_WilliamsR_1dElderRay_Trend_Volume_v1"
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
    open_ = prices['open'].values
    
    # Get 1d data for Elder Ray trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray (Bull Power and Bear Power)
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_1d = pd.Series(df_1d['close'])
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    ema_13 = close_1d.ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = (high_1d - ema_13).values
    bear_power = (low_1d - ema_13).values
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R oversold (< -80) for longs, overbought (> -20) for shorts
            williams_long = williams_r[i] < -80
            williams_short = williams_r[i] > -20
            
            # 1d Elder Ray trend filter: Bull Power > 0 for longs, Bear Power < 0 for shorts
            elder_long = bull_power_aligned[i] > 0
            elder_short = bear_power_aligned[i] < 0
            
            if williams_long and elder_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif williams_short and elder_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion complete) or Elder Ray turns negative
            if williams_r[i] > -50 or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion complete) or Elder Ray turns positive
            if williams_r[i] < -50 or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals