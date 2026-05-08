#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR filter
# Uses 20-period Donchian channels for structure, volume spike for confirmation,
# and ATR-based volatility filter to avoid choppy markets. Designed to work in
# both bull (breakouts continue) and bear (false breakouts filtered by volatility).
name = "4h_Donchian_Breakout_Volume_ATR_Filter"
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
    
    # 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # ATR for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop) and extreme volatility
        vol_filter = (atr[i] > 0.01 * close[i]) and (atr[i] < 0.05 * close[i])
        
        if position == 0:
            # Long: break above Donchian high + volume confirmation + volatility filter
            if (close[i] > donchian_high[i] and 
                vol_ratio[i] > 1.5 and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume confirmation + volatility filter
            elif (close[i] < donchian_low[i] and 
                  vol_ratio[i] > 1.5 and vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low or volatility too high
            if (close[i] < donchian_low[i] or 
                atr[i] > 0.05 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above Donchian high or volatility too high
            if (close[i] > donchian_high[i] or 
                atr[i] > 0.05 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals