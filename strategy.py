#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 4-hour volume confirmation and 1-day volatility filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period average + daily ATR(20) < 50-day SMA of ATR
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period average + daily ATR(20) < 50-day SMA of ATR
# Exit when price crosses 4-period EMA in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day ATR regime filter to avoid high volatility periods
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_4h_vol_1d_atr_regime_v1"
timeframe = "4h"
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
    
    # 1-day data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = tr1_1d[0]
    tr3_1d[0] = tr1_1d[0]
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 50-day SMA of ATR for regime filter
    atr_1d_s = pd.Series(atr_1d)
    atr_ma_50 = atr_1d_s.rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volatility regime: low volatility when current ATR < 50-day SMA of ATR
    vol_regime = atr_1d < atr_ma_50
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 4-period EMA for exit
    ema_4 = pd.Series(close).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_4[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 4-period EMA
            elif close[i] < ema_4[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 4-period EMA
            elif close[i] > ema_4[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and volatility regime
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # Volatility regime filter: low volatility (ATR < 50-day SMA of ATR)
            vol_filter = vol_regime_aligned[i]
            
            # Long: price breaks above Donchian high + volume filter + vol regime
            if close[i] > highest_high[i] and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + vol regime
            elif close[i] < lowest_low[i] and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals