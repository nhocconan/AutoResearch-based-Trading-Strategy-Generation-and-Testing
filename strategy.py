#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h KAMA for trend direction and 1d Bollinger Bands for mean-reversion entries.
# Long when price touches lower Bollinger Band with KAMA uptrend, short when price touches upper band with KAMA downtrend.
# Exit when price crosses KAMA or reaches opposite band. Uses 1.5x volume confirmation to avoid false signals.
# Designed to capture mean-reversion in ranging markets while filtering with higher timeframe trend.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for KAMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, 10))
    volatility = np.sum(np.abs(np.diff(close_12h, 1)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = np.power(er * (0.66 - 0.06) + 0.06, 2)
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # Start after 10 periods
    for i in range(10, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA slope for trend direction
    kama_slope = np.diff(kama, prepend=np.nan)
    
    # Load 1d data ONCE for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    kama_slope_aligned = align_htf_to_ltf(prices, df_12h, kama_slope)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need KAMA (30) and BB (20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(kama_slope_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: KAMA slope positive for uptrend, negative for downtrend
        uptrend = kama_slope_aligned[i] > 0
        downtrend = kama_slope_aligned[i] < 0
        
        if position == 0:
            # Look for Bollinger Band touches
            # Long: price touches lower band AND uptrend
            if (low[i] <= bb_lower_aligned[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price touches upper band AND downtrend
            elif (high[i] >= bb_upper_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses KAMA or reaches upper band
            if (close[i] >= kama_aligned[i] or 
                high[i] >= bb_upper_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses KAMA or reaches lower band
            if (close[i] <= kama_aligned[i] or 
                low[i] <= bb_lower_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_1dBB_MeanReversion_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0