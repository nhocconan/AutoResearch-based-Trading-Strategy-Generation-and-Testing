#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness regime + 1-day Donchian(20) breakout with volume confirmation
# Long when 12h Choppiness > 61.8 (ranging) + price breaks above 1d Donchian high + volume > 1.5x 20-period avg
# Short when 12h Choppiness > 61.8 (ranging) + price breaks below 1d Donchian low + volume > 1.5x 20-period avg
# Exit when price crosses 12h EMA(10) in opposite direction
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25
# Uses 12h Choppiness to filter trending markets (avoid whipsaw) and 1d Donchian/volume for entries
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_chop_regime_donchian_vol_v3"
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
    
    # 12h data for Choppiness and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1-day data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h Choppiness (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR
    atr_sum = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum / (hh_14 - ll_14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.zeros(len(atr_sum))
    for i in range(len(chop)):
        if range_14[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_14[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # 12h EMA(10) for exit
    ema_10 = pd.Series(close_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_12h, ema_10)
    
    # 1-day Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
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
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_10_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: price crosses below 12h EMA(10)
            elif close[i] < ema_10_aligned[i]:
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
            # Exit: price crosses above 12h EMA(10)
            elif close[i] > ema_10_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Chop > 61.8 (ranging) + Donchian breakout + volume confirmation
            chop_condition = chop_aligned[i] > 61.8
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: Chop > 61.8 + price breaks above Donchian high + volume filter
            if chop_condition and close[i] > highest_high[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Chop > 61.8 + price breaks below Donchian low + volume filter
            elif chop_condition and close[i] < lowest_low[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals