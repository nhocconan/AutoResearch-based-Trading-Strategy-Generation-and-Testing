# ICS_RandomWalk_LiquidityCapture_v1
# Hypothesis: Bitcoin and Ethereum exhibit asymmetric liquidity-seeking behavior where price moves toward
# zones of historical liquidity (unfilled gaps, equal highs/lows) before reversing. This strategy identifies
# liquidity zones on the daily timeframe and enters on 6-hour breaks toward these zones with volume
# confirmation, expecting mean reversion as liquidity is captured. Works in both bull/bear as liquidity
# grabs occur in all regimes. Targets 15-25 trades/year with disciplined exits.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for liquidity zones
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Identify daily liquidity zones: equal highs/lows and unfilled gaps
    # Equal highs: today's high == yesterday's high (within 0.1%)
    equal_high = np.zeros(len(high_1d), dtype=bool)
    equal_low = np.zeros(len(low_1d), dtype=bool)
    gap_up = np.zeros(len(high_1d), dtype=bool)  # today's low > yesterday's high
    gap_down = np.zeros(len(low_1d), dtype=bool)  # today's high < yesterday's low
    
    for i in range(1, len(high_1d)):
        # Equal highs/lows (liquidity pools)
        if abs(high_1d[i] - high_1d[i-1]) / high_1d[i-1] < 0.001:
            equal_high[i] = True
        if abs(low_1d[i] - low_1d[i-1]) / low_1d[i-1] < 0.001:
            equal_low[i] = True
        # Gaps (unfilled liquidity)
        if low_1d[i] > high_1d[i-1]:
            gap_up[i] = True
        if high_1d[i] < low_1d[i-1]:
            gap_down[i] = True
    
    # Combine liquidity signals: 1 = liquidity above (sell side), -1 = liquidity below (buy side)
    liquidity_signal = np.zeros(len(high_1d))
    liquidity_signal[equal_high | gap_up] = 1   # Liquidity above - potential sell zone
    liquidity_signal[equal_low | gap_down] = -1 # Liquidity below - potential buy zone
    
    # Forward fill liquidity signals (zones persist until filled)
    liquidity_signal = pd.Series(liquidity_signal).replace(0, np.nan).ffill().fillna(0).values
    
    # Align daily liquidity signal to 6h timeframe
    liquidity_6h = align_htf_to_ltf(prices, df_1d, liquidity_signal)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid extremely low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need volume MA20, ATR MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(liquidity_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        # Volatility filter: ATR > 0.5 * ATR MA20 (avoid extremely low volatility)
        volatility_filter = atr[i] > (0.5 * atr_ma20[i])
        
        if position == 0:
            # Long: liquidity below (buy side) AND price breaks down toward liquidity with volume
            if (liquidity_6h[i] == -1 and low[i] < close[i-1] and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: liquidity above (sell side) AND price breaks up toward liquidity with volume
            elif (liquidity_6h[i] == 1 and high[i] > close[i-1] and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: liquidity captured OR price breaks above recent high
            if liquidity_6h[i] != -1 or high[i] > np.maximum.reduce(high[max(0,i-5):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: liquidity captured OR price breaks below recent low
            if liquidity_6h[i] != 1 or low[i] < np.minimum.reduce(low[max(0,i-5):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "ICS_RandomWalk_LiquidityCapture_v1"
timeframe = "6h"
leverage = 1.0