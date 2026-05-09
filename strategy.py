#!/usr/bin/env python3
# 4H_4H_Close_VWAP_Bands_1D_Pullback_Strategy
# Uses 4h VWAP bands (1.5*ATR) and 1d close pullback
# Long: Price touches lower VWAP band AND price < 1d close (pullback in uptrend)
# Short: Price touches upper VWAP band AND price > 1d close (pullback in downtrend)
# Exit: Price crosses back to VWAP or 1d close direction reverses
# Position size: 0.25 (25%) to manage drawdown and reduce churn

name = "4H_4H_Close_VWAP_Bands_1D_Pullback_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den > 0, vwap_num / vwap_den, 0)
    
    # ATR for band width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # VWAP bands: ±1.5 * ATR
    upper_band = vwap + 1.5 * atr
    lower_band = vwap - 1.5 * atr
    
    # 1d close for pullback filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need enough data for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(close_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches lower VWAP band AND price < 1d close (pullback in uptrend)
            if (low[i] <= lower_band[i] and close[i] < close_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper VWAP band AND price > 1d close (pullback in downtrend)
            elif (high[i] >= upper_band[i] and close[i] > close_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back to VWAP OR 1d close direction turns down
            if (close[i] >= vwap[i]) or (close[i] >= close_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back to VWAP OR 1d close direction turns up
            if (close[i] <= vwap[i]) or (close[i] <= close_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals