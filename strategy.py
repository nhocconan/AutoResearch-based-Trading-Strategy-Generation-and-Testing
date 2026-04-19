#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily ATR filter and volume confirmation.
# Long when price breaks above Donchian high with volume > 1.5x average and ATR(14) > 0.5 * price.
# Short when price breaks below Donchian low with volume > 1.5x average and ATR(14) > 0.5 * price.
# Uses ATR filter to avoid ranging markets and volume to confirm breakout strength.
# Target: 12-30 trades/year per symbol (~48-120 total over 4 years).
name = "12h_Donchian20_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period) on 12h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 14)  # Donchian needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        atr_val = atr[i]
        
        # Volume and ATR confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        atr_confirmed = atr_val > 0.5 * price  # Significant volatility
        
        if position == 0:
            # Enter long: price breaks above Donchian high with volume and ATR confirmation
            if price > upper and volume_confirmed and atr_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with volume and ATR confirmation
            elif price < lower and volume_confirmed and atr_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below Donchian low
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above Donchian high
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals