#!/usr/bin/env python3
"""
4h_Keltner_Upper_Band_Touch_Volume_Mean_Reversion
Hypothesis: When price touches the upper Keltner Channel (EMA20 + 2*ATR10) with high volume,
it often signals overextension and mean reversion downward. Conversely, touching the lower
band with high volume signals oversold conditions and mean reversion upward. Works in both
bull and bear markets by fading extremes with volume confirmation.
Designed for 20-30 trades/year on 4h timeframe with low trade frequency to minimize fee drag.
"""

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
    
    # Keltner Channel: EMA20 +/- 2*ATR10
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR10: True Range average
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_band = ema_20 + 2.0 * atr_10
    lower_band = ema_20 - 2.0 * atr_10
    
    # Volume mean: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_high = volume > (1.5 * vol_ma)  # 1.5x volume for signal
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20[i]) or 
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price touches or crosses below lower band with high volume (oversold bounce)
            if price <= lower_band[i] and volume_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses above upper band with high volume (overbought fade)
            elif price >= upper_band[i] and volume_high[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit when price returns to EMA20 (mean reversion target)
            signals[i] = 0.25
            if price >= ema_20[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: exit when price returns to EMA20 (mean reversion target)
            signals[i] = -0.25
            if price <= ema_20[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Upper_Band_Touch_Volume_Mean_Reversion"
timeframe = "4h"
leverage = 1.0