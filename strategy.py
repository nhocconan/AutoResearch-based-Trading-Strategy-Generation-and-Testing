#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear power) with 12h EMA trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Go long when Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA(20) AND volume > 1.5x avg
# Go short when Bear Power < 0 AND Bull Power < 0 AND price < 12h EMA(20) AND volume > 1.5x avg
# Exit when power signals weaken or reverse
# Targets 50-150 trades over 4 years by requiring strong bull/bear power alignment + trend + volume

name = "6h_elder_ray_12h_ema_vol_v1"
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
    
    # Elder Ray: EMA(13) of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # 12h EMA(20) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bull Power <= 0 OR Bear Power >= 0 OR price < 12h EMA(20)
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power >= 0 OR Bear Power <= 0 OR price > 12h EMA(20)
            if bull_power[i] >= 0 or bear_power[i] <= 0 or close[i] > ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: strong bull/bear power + trend + volume
            if volume[i] > volume_threshold[i]:
                # Strong bullish: Bull Power > 0 AND Bear Power < 0 (bulls in control)
                if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_20_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Strong bearish: Bear Power < 0 AND Bull Power < 0 (bears in control)
                elif bear_power[i] < 0 and bull_power[i] < 0 and close[i] < ema_20_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals