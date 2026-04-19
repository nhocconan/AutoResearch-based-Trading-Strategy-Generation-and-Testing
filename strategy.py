#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above 20-period high with volume > 1.5x 20-period average.
# Short when price breaks below 20-period low with volume > 1.5x 20-period average.
# Exit on opposite Donchian break or ATR trailing stop.
# Uses discrete position sizes (0.30) to minimize churn. Designed for 4h timeframe
# to capture multi-day trends while avoiding whipsaws in both bull and bear markets.
# Target: 25-50 trades/year per symbol (~100-200 total over 4 years).
name = "4h_Donchian20_Volume_ATRStop"
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]  # First TR
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above Donchian high with volume confirmation
            if price > donchian_high[i] and volume_confirmed:
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
            # Enter short if price breaks below Donchian low with volume confirmation
            elif price < donchian_low[i] and volume_confirmed:
                signals[i] = -0.30
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            
            # Check for exit conditions
            # 1. Price breaks below Donchian low (opposite breakout)
            # 2. ATR trailing stop: price drops 2.5*ATR from highest since entry
            if price < donchian_low[i] or price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            
            # Check for exit conditions
            # 1. Price breaks above Donchian high (opposite breakout)
            # 2. ATR trailing stop: price rises 2.5*ATR from lowest since entry
            if price > donchian_high[i] or price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals