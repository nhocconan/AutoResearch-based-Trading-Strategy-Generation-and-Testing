#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and 1-day ATR volatility filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period average + ATR(14) > 0.5*ATR(50)
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period average + ATR(14) > 0.5*ATR(50)
# Exit when price crosses 12-period EMA in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day volume for confirmation and volatility regime filter to avoid choppy markets
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_vol_volatility_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12-period EMA for exit
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # ATR(14) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR(50) for volatility regime filter
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_12[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 12-period EMA
            elif close[i] < ema_12[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 12-period EMA
            elif close[i] > ema_12[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and volatility filter
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Volatility filter: ATR(14) > 0.5 * ATR(50) (ensures sufficient volatility)
            volatility_filter = atr_14[i] > 0.5 * atr_50[i]
            
            # Long: price breaks above Donchian high + volume filter + volatility filter
            if close[i] > highest_high[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + volatility filter
            elif close[i] < lowest_low[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals