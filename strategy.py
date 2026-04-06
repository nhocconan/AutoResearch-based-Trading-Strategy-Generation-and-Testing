#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_swingfailure_vol_volatility"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 20-period volume moving average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get 1d data for swing failure points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily swing failure points
    # Bullish SFP: price makes new low but closes above prior low
    # Bearish SFP: price makes new high but closes below prior high
    bull_sfp = np.full(len(close_1d), np.nan)
    bear_sfp = np.full(len(close_1d), np.nan)
    
    for i in range(2, len(close_1d)):
        # Bullish SFP: current low < previous low AND current close > previous low
        if low_1d[i] < low_1d[i-1] and close_1d[i] > low_1d[i-1]:
            bull_sfp[i] = low_1d[i-1]  # trigger level
        # Bearish SFP: current high > previous high AND current close < previous high
        if high_1d[i] > high_1d[i-1] and close_1d[i] < high_1d[i-1]:
            bear_sfp[i] = high_1d[i-1]  # trigger level
    
    # Align SFP levels to 6h timeframe
    bull_sfp_aligned = align_htf_to_ltf(prices, df_1d, bull_sfp)
    bear_sfp_aligned = align_htf_to_ltf(prices, df_1d, bear_sfp)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    start = max(30, 20)
    
    for i in range(start, n):
        if (np.isnan(atr[i]) or np.isnan(bull_sfp_aligned[i]) or 
            np.isnan(bear_sfp_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        if position == 1:  # long
            if (close[i] < entry_price - 2.5 * atr[i]) or \
               (not volume_filter and close[i] < entry_price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            if (close[i] > entry_price + 2.5 * atr[i]) or \
               (not volume_filter and close[i] > entry_price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Long: bullish SFP break with volume
            if not np.isnan(bull_sfp_aligned[i]) and \
               close[i] > bull_sfp_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish SFP break with volume
            elif not np.isnan(bear_sfp_aligned[i]) and \
                 close[i] < bear_sfp_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals