#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, and price > 1d EMA50 with volume > 1.5x average.
# Short when Bear Power < 0, Bull Power < 0, and price < 1d EMA50 with volume > 1.5x average.
# Exit when either power signal weakens or price crosses 1d EMA50.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via strong Bull Power and in bear markets via strong Bear Power.
# 6h timeframe balances trade frequency and responsiveness.

name = "6h_ElderRay_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0, Bear Power < 0, price > 1d EMA50, volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0, Bull Power < 0, price < 1d EMA50, volume confirmation
            elif (bear_power[i] < 0 and bull_power[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 OR Bear Power >= 0 OR price < 1d EMA50
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 OR Bull Power >= 0 OR price > 1d EMA50
            if (bear_power[i] >= 0 or bull_power[i] >= 0 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals