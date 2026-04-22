#!/usr/bin/env python3

"""
Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation.
Trade long when %R crosses above oversold (-80) during 12h uptrend with volume spike.
Trade short when %R crosses below overbought (-20) during 12h downtrend with volume spike.
Williams %R identifies exhaustion points in trends, while 12h EMA50 filters for trend alignment.
Volume spike confirms momentum behind the reversal. Designed for low trade frequency
(12-37 trades/year) by requiring three confirmations: %R extreme, trend alignment, and volume.
Works in both bull and bear markets by following the 12h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        wr_oversold = williams_r[i] > -80 and williams_r[i-1] <= -80  # Cross above -80
        wr_overbought = williams_r[i] < -20 and williams_r[i-1] >= -20  # Cross below -20
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: %R crosses above -80 + 12h uptrend + volume spike
            if wr_oversold and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: %R crosses below -20 + 12h downtrend + volume spike
            elif wr_overbought and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: %R crosses opposite extreme or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: %R crosses above -20 (overbought) or trend turns down
                if williams_r[i] >= -20 or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: %R crosses below -80 (oversold) or trend turns up
                if williams_r[i] <= -80 or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0