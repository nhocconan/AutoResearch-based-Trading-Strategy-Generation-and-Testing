#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(55) breakout with weekly volume confirmation and volatility filter
# Long when price breaks above 55-period Donchian high + volume > 1.5x 20-day average + ATR(14) < 0.5 * price
# Short when price breaks below 55-period Donchian low + volume > 1.5x 20-day average + ATR(14) < 0.5 * price
# Exit when price crosses 10-period EMA in opposite direction
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 20-day volume for confirmation and ATR for volatility filter to avoid choppy markets
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian55_vol_volfilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 55-period Donchian channels
    highest_high = pd.Series(high).rolling(window=55, min_periods=55).max().values
    lowest_low = pd.Series(low).rolling(window=55, min_periods=55).min().values
    
    # 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(55, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_10[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely volatile conditions
        vol_filter = atr[i] < 0.5 * close[i]
        
        # Volume confirmation: volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
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
            # Look for entries: Donchian breakout with volume confirmation and volatility filter
            # Long: price breaks above Donchian high + volume filter + vol filter
            if close[i] > highest_high[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + vol filter
            elif close[i] < lowest_low[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals