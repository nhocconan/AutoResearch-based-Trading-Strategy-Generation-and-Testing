#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR stoploss.
# Long when price breaks above 20-period high, volume > 1.5x 20-period average, ATR > 0.01*price.
# Short when price breaks below 20-period low, volume > 1.5x 20-period average, ATR > 0.01*price.
# Exit when price crosses 10-period moving average in opposite direction or ATR-based stop hit.
# Uses discrete position sizes (0.25) to minimize churn. Designed for 4h timeframe
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
    
    # ATR (14-period) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(ma_10[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        atr_val = atr[i]
        ma_10_val = ma_10[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Volatility filter: avoid low-volatility environments
        volatility_filter = atr_val > 0.01 * price
        
        if position == 0:
            # Enter long if price breaks above Donchian high, volume confirmation, and volatility filter
            if price > donchian_high[i] and volume_confirmed and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below Donchian low, volume confirmation, and volatility filter
            elif price < donchian_low[i] and volume_confirmed and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below 10-period MA or ATR-based stop hit
            if price < ma_10_val or price < donchian_low[i]:  # Stop at Donchian low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above 10-period MA or ATR-based stop hit
            if price > ma_10_val or price > donchian_high[i]:  # Stop at Donchian high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals