#!/usr/bin/env python3
# 4h_donchian_volatility_breakout_v1
# Hypothesis: 4h strategy using Donchian(20) breakout with volatility expansion filter and volume confirmation.
# Enters long when price breaks above 4h Donchian(20) upper band, ATR(14) > 1.2x ATR(50), and volume > 1.3x 20-period average.
# Enters short when price breaks below 4h Donchian(20) lower band, ATR(14) > 1.2x ATR(50), and volume > 1.3x average.
# Uses discrete position sizing (±0.25) to minimize fee churn.
# Volatility filter ensures entries occur during expansion phases, reducing whipsaws in ranging markets.
# Target: 75-150 total trades over 4 years (19-38/year). Works in bull/bear via Donchian structure and volatility filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.insert(tr1, 0, high[0] - low[0])  # first TR
    atr14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr1).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_expansion = atr14 > (atr50 * 1.2)  # ATR(14) > 1.2x ATR(50)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.3)  # Volume at least 1.3x average
    
    # Calculate Donchian channels for 4h (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 4h Donchian lower band
            if close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above 4h Donchian upper band
            if close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 4h Donchian upper, volatility expansion, volume spike
            if (close[i] > donchian_upper[i]) and vol_expansion[i] and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 4h Donchian lower, volatility expansion, volume spike
            elif (close[i] < donchian_lower[i]) and vol_expansion[i] and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals