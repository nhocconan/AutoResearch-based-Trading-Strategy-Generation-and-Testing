#!/usr/bin/env python3
# 6h_OrderFlow_Imbalance_Pullback
# Hypothesis: Combines order flow imbalance detection with pullback entries in 6h timeframe.
# Uses volume delta (buy vs sell volume) to detect institutional accumulation/distribution.
# Enters on pullbacks to VWAP during strong order flow imbalances, filtered by 12h trend.
# Works in bull markets (accumulation on dips) and bear markets (distribution on rallies).
# Target: 15-25 trades/year per symbol with strict entry criteria to minimize fee drag.

name = "6h_OrderFlow_Imbalance_Pullback"
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
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # Calculate volume delta (buying pressure - selling pressure)
    # taker_buy_volume = volume * (buy side)
    # volume - taker_buy_volume = volume * (sell side)
    volume_delta = 2 * taker_buy_volume - volume  # buy_volume - sell_volume
    
    # Calculate VWAP for 6h timeframe
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    # Cumulative VWAP with reset conditions (reset when price gaps significantly)
    cum_pv = np.cumsum(pv)
    cum_volume = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_volume, out=np.full_like(cum_pv, np.nan), where=cum_volume!=0)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 49) / 50
    
    # Align 12h indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume delta momentum (3-period sum)
    volume_delta_momentum = np.full_like(volume_delta, np.nan)
    if len(volume_delta) >= 3:
        for i in range(2, len(volume_delta)):
            volume_delta_momentum[i] = volume_delta[i] + volume_delta[i-1] + volume_delta[i-2]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_delta_momentum[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        uptrend_12h = close[i] > ema_50_12h_aligned[i]
        downtrend_12h = close[i] < ema_50_12h_aligned[i]
        
        # Strong order flow imbalance thresholds
        strong_buying = volume_delta_momentum[i] > 0 and volume_delta[i] > 0
        strong_selling = volume_delta_momentum[i] < 0 and volume_delta[i] < 0
        
        if position == 0:
            # Enter long: Pullback to VWAP during strong buying pressure in uptrend
            if (uptrend_12h and strong_buying and 
                low[i] <= vwap[i] <= high[i]):  # Price touches VWAP
                signals[i] = 0.25
                position = 1
            # Enter short: Pullback to VWAP during strong selling pressure in downtrend
            elif (downtrend_12h and strong_selling and 
                  low[i] <= vwap[i] <= high[i]):  # Price touches VWAP
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price moves above VWAP significantly or order flow turns negative
            if (close[i] > vwap[i] * 1.01 or  # 1% above VWAP
                volume_delta[i] < -volume[i] * 0.1):  # Significant selling pressure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price moves below VWAP significantly or order flow turns positive
            if (close[i] < vwap[i] * 0.99 or  # 1% below VWAP
                volume_delta[i] > volume[i] * 0.1):  # Significant buying pressure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals