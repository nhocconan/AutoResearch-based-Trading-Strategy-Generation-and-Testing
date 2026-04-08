#!/usr/bin/env python3
# 12h_camilla_pivot_breakout_volume_v2
# Hypothesis: Uses 1d Camarilla pivot levels (H3/L3) as entry triggers with volume confirmation and 12h trend filter.
# Long when price breaks above H3 with volume surge and 12h close > 12h open (bullish candle).
# Short when price breaks below L3 with volume surge and 12h close < 12h open (bearish candle).
# Exit when price returns to the 12h VWAP or opposite pivot level is touched.
# Designed for low-frequency, high-conviction trades in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 24-period average (24 * 12h = 12 days)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    H3 = np.full(len(close_1d), np.nan)
    L3 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(high_1d[i]) and not np.isnan(low_1d[i]) and not np.isnan(close_1d[i]):
            range_1d = high_1d[i] - low_1d[i]
            close_1d_i = close_1d[i]
            H3[i] = close_1d_i + 1.1 * range_1d / 6
            L3[i] = close_1d_i - 1.1 * range_1d / 6
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get 12h VWAP for exit condition
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, out=np.full_like(vwap_numerator, np.nan), where=vwap_denominator!=0)
    
    # Get 12h open/close for trend filter (bullish/bearish candle)
    open_12h = prices['open'].values
    close_12h = close  # already have close
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1)  # Need volume MA and at least one prior bar
    
    for i in range(start_idx, n):
        # Skip if required data not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to VWAP or touches L3 (opposite pivot)
            if close[i] <= vwap[i] or close[i] <= L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to VWAP or touches H3 (opposite pivot)
            if close[i] >= vwap[i] or close[i] >= H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above H3 with volume surge and bullish 12h candle
            if (close[i] > H3_aligned[i] and 
                vol_surge[i] and 
                close_12h[i] > open_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 with volume surge and bearish 12h candle
            elif (close[i] < L3_aligned[i] and 
                  vol_surge[i] and 
                  close_12h[i] < open_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals