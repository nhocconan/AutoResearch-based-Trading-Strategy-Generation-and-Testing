#!/usr/bin/env python3
# 4H_1D_VWAP_Touch_Reversal_With_Volume_And_Trend
# Hypothesis: On 4h timeframe, enter long when price touches VWAP from below with bullish 1d trend and volume confirmation, and short when price touches VWAP from above with bearish 1d trend and volume confirmation.
# Uses VWAP as dynamic support/resistance, which adapts to price action and volume.
# In ranging markets, VWAP acts as mean reversion zone; in trends, it acts as dynamic support/resistance.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years).

name = "4H_1D_VWAP_Touch_Reversal_With_Volume_And_Trend"
timeframe = "4h"
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
    
    # Get 1d data for VWAP and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP for 1d: cumulative (price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d indicators to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches VWAP from below (close crosses above VWAP) + 1d uptrend + volume confirmation
            if close[i] > vwap_1d_aligned[i] and close[i-1] <= vwap_1d_aligned[i-1] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches VWAP from above (close crosses below VWAP) + 1d downtrend + volume confirmation
            elif close[i] < vwap_1d_aligned[i] and close[i-1] >= vwap_1d_aligned[i-1] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP (reversal) or trend changes to down
            if close[i] < vwap_1d_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP (reversal) or trend changes to up
            if close[i] > vwap_1d_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals