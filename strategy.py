#!/usr/bin/env python3
"""
4h_Keltner_Channel_20_MeanReversion
Strategy: Mean reversion at Keltner Channel bands with volume confirmation and 1D trend filter.
- Long when price touches lower Keltner Channel (20, 2.0) + volume > 1.5x 20-period avg + 1D close > 1D EMA50
- Short when price touches upper Keltner Channel (20, 2.0) + volume > 1.5x 20-period avg + 1D close < 1D EMA50
- Exit when price returns to 20-period EMA or opposite touch occurs
- Position size: ±0.25
- Uses 4h timeframe as primary with 1D for trend filter
"""

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
    
    # Calculate EMA20 for Keltner middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(20) for Keltner width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    kc_upper = ema20 + (2.0 * atr20)
    kc_lower = ema20 - (2.0 * atr20)
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1D EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 50)  # EMA20, ATR20, volume MA20, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Keltner Channel touch conditions
        touch_lower = low[i] <= kc_lower[i]  # touch or penetrate lower band
        touch_upper = high[i] >= kc_upper[i]  # touch or penetrate upper band
        
        # Return to EMA20 for exit
        return_to_ema = abs(close[i] - ema20[i]) < (0.001 * ema20[i])  # within 0.1% of EMA20
        
        if position == 0:
            # Long: touch lower band + volume filter + 1D uptrend
            if touch_lower and volume_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: touch upper band + volume filter + 1D downtrend
            elif touch_upper and volume_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to EMA20 or opposite touch
            if return_to_ema or touch_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to EMA20 or opposite touch
            if return_to_ema or touch_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Channel_20_MeanReversion"
timeframe = "4h"
leverage = 1.0