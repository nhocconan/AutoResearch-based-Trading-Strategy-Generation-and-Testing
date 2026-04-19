#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d volume confirmation and volatility filter
# Uses daily Camarilla levels (H4, L4, H3, L3) as entry triggers on 12h timeframe
# Volume filter: current volume > 1.5x 20-period average
# Volatility filter: ATR(14) > 0.5 * ATR(50) to avoid low-volatility chop
# Trend filter: 1d EMA50 (long above, short below)
# Designed for 12h timeframe with target of 15-25 trades/year to minimize fee drag
# Works in both bull (breakouts above H3/H4) and bear (breakdowns below L3/L4) markets
name = "12h_Camarilla_Pivot_Breakout_Volume_Volatility_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    # Based on previous day's high, low, close
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Calculate pivot and ranges
    pivot = (phigh + plow + pclose) / 3.0
    range_val = phigh - plow
    
    # Camarilla levels
    h4 = pclose + range_val * 1.1 / 2
    h3 = pclose + range_val * 1.1 / 4
    l3 = pclose - range_val * 1.1 / 4
    l4 = pclose - range_val * 1.1 / 2
    
    # Align to 12h timeframe (wait for daily bar to close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h ATR for volatility filter and stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volume filter
    if n >= 20:
        avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    else:
        avg_volume = volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > 1.5 * avg_volume[i]
        
        # Volatility filter: ATR(14) > 0.5 * ATR(50) to avoid low-volatility chop
        volatility_filter = atr_14[i] > 0.5 * atr_50[i]
        
        if position == 0:
            # Long: Price breaks above H3 or H4 with volume and volatility, and above 1d EMA50
            if (close[i] > h3_aligned[i] or close[i] > h4_aligned[i]) and volume_filter and volatility_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 or L4 with volume and volatility, and below 1d EMA50
            elif (close[i] < l3_aligned[i] or close[i] < l4_aligned[i]) and volume_filter and volatility_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price retracement to L3 or 2x ATR stop
            if close[i] < l3_aligned[i] or close[i] < close[i-1] - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price retracement to H3 or 2x ATR stop
            if close[i] > h3_aligned[i] or close[i] > close[i-1] + 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals