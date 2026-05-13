#!/usr/bin/env python3
"""
6h_Parabolic_SAR_Trend_With_Volume_Filter
Hypothesis: Parabolic SAR provides dynamic stop-and-reverse levels that work well in trending markets.
Combined with volume confirmation and daily trend filter, it captures sustained moves while avoiding whipsaws.
In bear markets, the SAR quickly reverses on downturns; in bull, it trails uptrends. Volume filter ensures
only institutional participation drives entries. Target: 20-50 trades/year on 6B timeframe.
"""

name = "6h_Parabolic_SAR_Trend_With_Volume_Filter"
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
    
    # Parabolic SAR calculation
    def parabolic_sar(high, low, start=0.02, increment=0.02, maximum=0.2):
        n = len(high)
        sar = np.zeros(n)
        trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
        af = start
        ep = low[0]  # extreme point
        
        sar[0] = low[0]
        trend[0] = 1
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                # SAR cannot exceed previous two lows
                if i >= 2:
                    sar[i] = min(sar[i], low[i-1], low[i-2])
                # Check for trend reversal
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep
                    af = start
                    ep = high[i]
                else:
                    trend[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + increment, maximum)
            else:  # downtrend
                sar[i] = sar[i-1] + af * (sar[i-1] - ep)
                # SAR cannot be below previous two highs
                if i >= 2:
                    sar[i] = max(sar[i], high[i-1], high[i-2])
                # Check for trend reversal
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep
                    af = start
                    ep = low[i]
                else:
                    trend[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + increment, maximum)
        
        return sar
    
    sar = parabolic_sar(high, low)
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price above SAR (uptrend) with volume confirmation and daily uptrend
            if (close[i] > sar[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below SAR (downtrend) with volume confirmation and daily downtrend
            elif (close[i] < sar[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below SAR (trend reversal)
            if close[i] < sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above SAR (trend reversal)
            if close[i] > sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals