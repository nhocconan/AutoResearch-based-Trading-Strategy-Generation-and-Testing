#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d CAMARILLA pivot levels with volume confirmation and 1w trend filter.
# Long when price breaks above H3 level with 1w RSI > 50 and volume > 1.5x average.
# Short when price breaks below L3 level with 1w RSI < 50 and volume > 1.5x average.
# Exit when price returns to PIVOT level or RSI crosses 50 in opposite direction.
# Designed to work in both bull and bear markets by using pivot levels (support/resistance) 
# and RSI for trend confirmation. Target: 15-25 trades/year per symbol (60-100 total over 4 years)
# to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for CAMARILLA pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CAMARILLA pivot levels for prior day
    # Pivot = (H + L + C) / 3
    # H3 = C + (H - L) * 1.1/2
    # L3 = C - (H - L) * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = close_1d + range_1d * 1.1 / 2.0
    l3_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Load 1w data ONCE for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on 1w
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need volume MA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: RSI > 50 for uptrend, < 50 for downtrend
        uptrend = rsi_1w_aligned[i] > 50
        downtrend = rsi_1w_aligned[i] < 50
        
        if position == 0:
            # Look for CAMARILLA breakouts
            # Long: price breaks above H3 level AND uptrend
            if (close[i] > h3_aligned[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below L3 level AND downtrend
            elif (close[i] < l3_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to PIVOT level or RSI crosses below 50
            if (close[i] <= pivot_aligned[i] or 
                rsi_1w_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to PIVOT level or RSI crosses above 50
            if (close[i] >= pivot_aligned[i] or 
                rsi_1w_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_H3L3_Pivot_1wRSI_v1"
timeframe = "12h"
leverage = 1.0