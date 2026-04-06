#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Enter long when: Bull Power > 0, Bear Power < 0 (bullish divergence), 1d EMA(50) uptrend, volume > 1.5x avg
# Enter short when: Bear Power < 0, Bull Power > 0 (bearish divergence), 1d EMA(50) downtrend, volume > 1.5x avg
# Exit when: Elder Ray divergence breaks OR price crosses EMA(13)
# Target: 50-150 trades over 4 years by requiring multiple confluence factors
# Works in bull (captures momentum) and bear (avoids counter-trend via 1d filter)

name = "6h_elder_ray_1dtrend_vol_v1"
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
    
    # Elder Ray components: EMA(13) of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Elder Ray turns bearish OR price closes below EMA13
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Elder Ray turns bullish OR price closes above EMA13
            if bull_power[i] >= 0 or bear_power[i] <= 0 or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray divergence + 1d trend + volume
            if volume[i] > volume_threshold[i]:
                # Bullish: Bull Power > 0 (buying pressure) AND Bear Power < 0 (weak selling)
                if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Bear Power < 0 (selling pressure) AND Bull Power > 0 (weak buying)
                elif bear_power[i] < 0 and bull_power[i] > 0 and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals