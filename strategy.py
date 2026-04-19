#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band width breakout with 10-day ATR volatility filter and volume confirmation.
# BB width (upper - lower) / middle band measures volatility contraction/expansion.
# Expansion breakout with high volume captures trending moves in both bull and bear markets.
# Bollinger Bands use 20-period SMA with 2 standard deviations.
# Filters: BB width expansion > 1.3x 20-period average, price breaks above upper BB (long) or below lower BB (short),
# volume > 1.8x 20-period average. Exit when price crosses back inside Bollinger Bands.
# Works in bull (breakout continuation) and bear (breakdown continuation). Target: 20-30 trades/year per symbol.
name = "4h_BollingerWidthBreakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = upper_bb - lower_bb
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for BB and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        bb_w = bb_width[i]
        bb_w_ma = bb_width_ma[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        
        if position == 0:
            # Breakout entry: BB width expansion + volume spike + price breaks BB
            if (bb_w > 1.3 * bb_w_ma and vol > 1.8 * vol_ma):
                if price > upper:  # Break above upper band
                    signals[i] = 0.25
                    position = 1
                elif price < lower:  # Break below lower band
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside Bollinger Bands (below upper band)
            if price < upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside Bollinger Bands (above lower band)
            if price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals