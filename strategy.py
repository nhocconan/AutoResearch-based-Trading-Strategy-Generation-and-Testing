#!/usr/bin/env python3
# 4h_VWAP_Rewave_Pullback
# Hypothesis: Pullbacks to volume-weighted average price (VWAP) with trend confirmation
# using weekly EMA200. Long when price pulls back to VWAP in uptrend, short when
# price rallies to VWAP in downtrend. Uses 4h VWAP and weekly EMA200 for trend filter.
# Entry conditions: price crosses VWAP with volume above average and trend alignment.
# Exit: price moves 1.5x ATR away from VWAP or trend reversal.
# Designed for low trade frequency (<50/year) to minimize fee drag in ranging/bear markets.

name = "4h_VWAP_Rewave_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 4h VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_volume = np.nancumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_volume, out=np.full_like(cum_pv, np.nan), where=cum_volume!=0)
    
    # Calculate ATR for volatility filtering and exit
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: volume above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 200)  # Ensure enough data for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Trend determination from weekly EMA200
        uptrend = close[i] > ema200_1w_aligned[i]
        downtrend = close[i] < ema200_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 0:
            # Long: price crosses above VWAP in uptrend with volume
            if uptrend and vol_ok and close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP in downtrend with volume
            elif downtrend and vol_ok and close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price moves 1.5*ATR away from VWAP or trend reverses
            if (close[i] < vwap[i] - 1.5 * atr[i] or 
                not uptrend or 
                close[i] < vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price moves 1.5*ATR away from VWAP or trend reverses
            if (close[i] > vwap[i] + 1.5 * atr[i] or 
                not downtrend or 
                close[i] > vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals