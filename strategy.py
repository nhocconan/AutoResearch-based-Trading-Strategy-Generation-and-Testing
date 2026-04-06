#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA(50) trend filter + volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND price > 12h EMA(50) AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND price < 12h EMA(50) AND volume > 1.5x 20-period average
# Exit when Elder Power crosses zero (Bull Power < 0 for long exit, Bear Power > 0 for short exit)
# Uses 6h timeframe to balance signal frequency and noise, Elder Ray for momentum,
# 12h EMA for trend filter, volume for confirmation. Works in bull (trend following) and bear (mean reversion via power shifts).

name = "6h_elder_ray_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray: EMA(13) of close
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema13
    
    # 12-hour EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Elder Power crosses zero
        if position == 1:  # long position
            if bull_power[i] < 0:  # Bull Power turned negative
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bear_power[i] > 0:  # Bear Power turned positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power > 0 AND price > 12h EMA(50) AND volume confirmation
            if (bull_power[i] > 0 and close[i] > ema50_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price < 12h EMA(50) AND volume confirmation
            elif (bear_power[i] < 0 and close[i] < ema50_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals