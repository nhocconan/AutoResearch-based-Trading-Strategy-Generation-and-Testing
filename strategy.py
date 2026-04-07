# 12h_volatility_breakout_atr_v1
# Hypothesis: ATR-based volatility breakout on 12h with volume confirmation.
# Uses ATR(14) to detect volatility expansion (ATR > 1.2x 50-period MA).
# Breakout occurs when price closes beyond ATR-based bands (close ± 2*ATR).
# Volume confirmation requires volume > 1.5x 20-period MA.
# Volatility breakouts capture momentum bursts in both bull and bear markets.
# Low trade frequency expected due to strict volatility + volume + breakout confluence.
# Position size: 0.25 when conditions met, scaled by volatility regime (0.5-1.5x).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_breakout_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility measurement and breakout bands
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility expansion: ATR > 1.2x 50-period MA
    vol_expansion = atr > (1.2 * atr_ma)
    
    # ATR-based breakout bands: close ± 2*ATR
    upper_band = close + (2.0 * atr)
    lower_band = close - (2.0 * atr)
    
    # Breakout conditions
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(atr_ma[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime scaling: scale position size inversely with volatility
        vol_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        vol_scale = np.clip(1.0 / vol_ratio, 0.5, 1.5)  # scale between 0.5 and 1.5
        base_size = 0.25
        
        # Entry conditions: volatility expansion + breakout + volume confirmation
        if vol_expansion[i] and vol_confirm[i]:
            if breakout_up[i]:
                signals[i] = base_size * vol_scale  # long
            elif breakout_down[i]:
                signals[i] = -base_size * vol_scale  # short
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals