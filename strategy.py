#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with daily ATR filter and volume confirmation.
# Long when price breaks above 20-period high AND ATR(14) > 1.2x ATR(50) AND volume > 1.5x daily average volume
# Short when price breaks below 20-period low AND ATR(14) > 1.2x ATR(50) AND volume > 1.5x daily average volume
# Exit when price crosses back below/above 10-period moving average
# Uses Donchian for breakout structure, ATR ratio for volatility expansion, volume for confirmation.
# Target: 15-35 trades/year per symbol.
name = "6h_Donchian_Breakout_ATR_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Exit MA (10-period)
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # ATR calculation (14 and 50 periods)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Daily average volume (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure Donchian and ATR(50) are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ma_10[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        ma = ma_10[i]
        atr14 = atr_14[i]
        atr50 = atr_50[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # ATR filter: volatility expansion
        atr_expansion = atr14 > 1.2 * atr50
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: break above upper channel + ATR expansion + volume confirmation
            if price > upper_channel and atr_expansion and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower channel + ATR expansion + volume confirmation
            elif price < lower_channel and atr_expansion and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 10-period MA
            if price < ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-period MA
            if price > ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals