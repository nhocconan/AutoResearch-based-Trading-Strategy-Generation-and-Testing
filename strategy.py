#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and volatility filter
# Long when price breaks above 20-period high + volume > 1.5x average + ATR ratio > 0.8
# Short when price breaks below 20-period low + volume > 1.5x average + ATR ratio > 0.8
# Exit when price crosses opposite 10-period EMA
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses volume to confirm breakout strength and ATR ratio to filter low volatility
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_vol_atr_ratio_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR(14) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio (current ATR / 50-period ATR average) for volatility filter
    atr_s = pd.Series(atr)
    atr_ma = atr_s.rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_10[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 10-period EMA
            elif close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 10-period EMA
            elif close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and volatility filters
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # Volatility filter: ATR ratio > 0.8 (avoid low volatility periods)
            vol_filter = atr_ratio[i] > 0.8
            
            # Long: price breaks above Donchian high + volume filter + volatility filter
            if close[i] > highest_high[i] and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + volatility filter
            elif close[i] < lowest_low[i] and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals