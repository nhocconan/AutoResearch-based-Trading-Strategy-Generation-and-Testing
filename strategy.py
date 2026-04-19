#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA200 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 12h EMA200 AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 12h EMA200 AND volume > 1.5x average
# Uses discrete positions (0.25) to minimize churn. Donchian provides clear structure,
# EMA200 filters trend direction, volume confirms breakout strength.
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years).
name = "4h_Donchian20_EMA200_Volume"
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
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA200 on 12h
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA200 to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above Donchian high, above EMA200, and volume confirmation
            if price > donch_high and price > ema_200_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below Donchian low, below EMA200, and volume confirmation
            elif price < donch_low and price < ema_200_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below Donchian low or trend reverses
            if price < donch_low or price < ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above Donchian high or trend reverses
            if price > donch_high or price > ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals