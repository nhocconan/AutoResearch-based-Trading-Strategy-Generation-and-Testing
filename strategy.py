#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high and 1-day EMA(50) > EMA(200) (uptrend).
# Short when price breaks below Donchian(20) low and 1-day EMA(50) < EMA(200) (downtrend).
# Exit on opposite Donchian break or 2.5 * ATR stoploss.
# Volume confirmation: current volume > 1.8 * 20-period average volume.
# Position size: 0.28. Designed to work in both bull and bear markets by using daily trend filter.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_donchian20_1d_trend_vol_v2"
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
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) and EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.28
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian lower(20)
            elif close[i] < low[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian upper(20)
            elif close[i] > high[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.28
        else:
            # Calculate Donchian channels (20-period)
            highest_high = high[i-20:i].max() if i >= 20 else high[:i].max()
            lowest_low = low[i-20:i].min() if i >= 20 else low[:i].min()
            
            # Trend filter: 1-day EMA(50) > EMA(200) for uptrend, < for downtrend
            uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
            downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.8 * average volume
            volume_confirm = volume[i] > 1.8 * vol_avg[i]
            
            # Long: price breaks above Donchian upper(20) in uptrend with volume
            if close[i] > highest_high and uptrend and volume_confirm:
                signals[i] = 0.28
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower(20) in downtrend with volume
            elif close[i] < lowest_low and downtrend and volume_confirm:
                signals[i] = -0.28
                position = -1
                entry_price = close[i]
    
    return signals