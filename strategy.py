#!/usr/bin/env python3
"""
4h_Keltner_Band_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Keltner bands (ATR-based) with volume spike and EMA trend filter on 4h timeframe.
Uses 20-period EMA for trend direction and 1.5x ATR for band width to capture breakouts in both bull/bear markets.
Target: 20-40 trades/year to minimize fee drift while capturing strong directional moves with proper risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA20 for trend direction
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for Keltner bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner bands: EMA20 ± 1.5 * ATR
    upper_band = ema_20 + (1.5 * atr)
    lower_band = ema_20 - (1.5 * atr)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema20 = ema_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and uptrend
            if price > upper and vol_spike and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike and downtrend
            elif price < lower and vol_spike and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below EMA20 OR touches opposite band
            if price < ema20:
                signals[i] = 0.0
                position = 0
            elif price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above EMA20 OR touches opposite band
            if price > ema20:
                signals[i] = 0.0
                position = 0
            elif price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Band_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0