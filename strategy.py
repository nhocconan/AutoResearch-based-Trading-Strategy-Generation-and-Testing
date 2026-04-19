#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume and volatility filter
# In trending markets, price breaks Donchian(20) high/low with volume expansion
# In ranging markets, price respects Donchian boundaries as support/resistance
# Volume confirms breakout authenticity, ATR filter avoids low-volatility false breakouts
# Works in bull/bear by adapting to volatility regime via ATR threshold
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)

name = "4h_DonchianBreakout_VolumeVolFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR(14) > 0.5 * 20-period ATR average
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > 0.5 * atr_ma_20  # Avoid low-volatility false breakouts
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        atr_ma = atr_ma_20[i]
        
        # Volume and volatility filters
        volume_confirmed = vol > 1.5 * vol_ma
        volatility_filter = vol_filter[i]
        
        # Donchian levels
        upper = highest_20[i]
        lower = lowest_20[i]
        
        if position == 0:
            # Long: break above upper band with volume and volatility
            if price > upper and volume_confirmed and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume and volatility
            elif price < lower and volume_confirmed and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle of channel or breaks lower band
            mid = (upper + lower) / 2.0
            if price < mid or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle of channel or breaks upper band
            mid = (upper + lower) / 2.0
            if price > mid or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals