#133789: 4h Donchian breakout + volume spike + RSI filter (long/short)
# Hypothesis: Donchian breakouts capture trends, volume spike confirms strength, RSI filters overextension.
# Works in bull (breakouts up) and bear (breakouts down). Target: 20-50 trades/year.

#!/usr/bin/env python3
name = "4h_Donchian_Breakout_VolumeRSI"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma20)
    
    # RSI(14) for overextension filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(vol_ma20[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper Donchian + volume spike + RSI not overbought (<70)
            if (close[i] > high_max[i] and 
                vol_spike[i] and 
                rsi[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian + volume spike + RSI not oversold (>30)
            elif (close[i] < low_min[i] and 
                  vol_spike[i] and 
                  rsi[i] > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below lower Donchian (reversal signal)
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above upper Donchian (reversal signal)
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals