#!/usr/bin/env python3
"""
#100939 - 6h_VolumePriceTrend_12hTrend_MeanReversion
Hypothesis: On 6b timeframe, combine volume-price trend (VPT) with 12h trend filter and mean reversion to VWAP.
In bull markets: VPT rising + price above 12h EMA50 triggers long entries.
In bear markets: VPT falling + price below 12h EMA50 triggers short entries.
Mean reversion exits when price crosses 6h VWAP.
Volume confirms institutional participation. Trend filter avoids counter-trend trades.
Targets 12-37 trades/year (50-150 total) to minimize fee drag.
Works in both bull (trend continuation) and bear (mean reversion within trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate VPT (Volume Price Trend) - cumulative volume * price change
    price_change = np.diff(close, prepend=close[0])
    vpt = np.cumsum(volume * price_change / close)
    
    # Calculate VWAP (Volume Weighted Average Price) for mean reversion exit
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, out=np.zeros_like(vwap_numerator), where=vwap_denominator!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period for VPT and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vpt[i]) or 
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: VPT rising (current > previous) AND price above 12h EMA50
        if (i > 0 and vpt[i] > vpt[i-1] and 
            close[i] > ema50_12h_aligned[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: VPT falling (current < previous) AND price below 12h EMA50
        elif (i > 0 and vpt[i] < vpt[i-1] and 
              close[i] < ema50_12h_aligned[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses 6h VWAP (mean reversion)
        elif position == 1 and close[i] < vwap[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > vwap[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_VolumePriceTrend_12hTrend_MeanReversion"
timeframe = "6h"
leverage = 1.0