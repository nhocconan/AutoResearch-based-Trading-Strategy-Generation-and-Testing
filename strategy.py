#!/usr/bin/env python3
# 6H_Keltner_Band_Breakout_12hTrend_VolumeFilter
# Hypothesis: Keltner Channel breakouts on 6h timeframe with 12h trend alignment and volume confirmation.
# In bull markets, price breaks above upper Keltner band with 12h uptrend and volume > 2x average.
# In bear markets, price breaks below lower Keltner band with 12h downtrend and volume > 2x average.
# Uses ATR-based channels that adapt to volatility, reducing false breakouts in ranging markets.
# Targets 12-37 trades/year (50-150 total over 4 years) on BTC, ETH, SOL.

name = "6H_Keltner_Band_Breakout_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Keltner Channel parameters
    atr_length = 10
    kc_multiplier = 2.0
    ema_length = 20
    
    # Calculate ATR, EMA, and Keltner Bands
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=atr_length, adjust=False, min_periods=atr_length).mean().values
    
    ema = pd.Series(close).ewm(span=ema_length, adjust=False, min_periods=ema_length).mean().values
    
    upper_keltner = ema + (kc_multiplier * atr)
    lower_keltner = ema - (kc_multiplier * atr)
    
    # 12h trend filter: EMA 50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, ema_length, atr_length)  # Ensure enough history
    
    for i in range(start_idx, n):
        if np.isnan(ema[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        is_uptrend = close[i] > ema_50_12h_aligned[i]
        is_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above upper Keltner band + volume confirmation + 12h uptrend
            if (close[i] > upper_keltner[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower Keltner band + volume confirmation + 12h downtrend
            elif (close[i] < lower_keltner[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below EMA (middle of Keltner Channel)
            if close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above EMA (middle of Keltner Channel)
            if close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals