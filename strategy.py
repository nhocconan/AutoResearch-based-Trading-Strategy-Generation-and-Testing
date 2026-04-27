#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with weekly trend filter and volume confirmation.
# Long when price breaks above upper BB with volume spike and weekly uptrend.
# Short when price breaks below lower BB with volume spike and weekly downtrend.
# Bollinger Band squeeze identifies low volatility periods that precede breakouts.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation validates institutional participation.
# Target: 15-25 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = ma + (bb_std * std)
    lower_bb = ma - (bb_std * std)
    
    # Bollinger Band width for squeeze detection
    bb_width = (upper_bb - lower_bb) / ma
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < bb_width_ma  # Width below its 50-period average
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on weekly close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above upper BB, squeeze, weekly uptrend, volume spike
        if (close[i] > upper_bb[i] and 
            squeeze_condition[i] and 
            close[i] > ema34_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below lower BB, squeeze, weekly downtrend, volume spike
        elif (close[i] < lower_bb[i] and 
              squeeze_condition[i] and 
              close[i] < ema34_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_BollingerSqueeze_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0