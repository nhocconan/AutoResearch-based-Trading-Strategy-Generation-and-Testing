#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily volume confirmation and 1-day EMA(50) trend filter
# Long when price breaks above 12h Donchian high + daily volume > 1.5x 20-period daily average + 12h close > 1d EMA(50)
# Short when price breaks below 12h Donchian low + daily volume > 1.5x 20-period daily average + 12h close < 1d EMA(50)
# Exit when price crosses opposite Donchian level
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily volume for confirmation and daily EMA(50) for trend filter
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_vol_ema50_v1"
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
    
    # 1-day data for volume and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50 = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: price crosses below Donchian low
            elif close[i] < lowest_low[i]:
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
            # Exit: price crosses above Donchian high
            elif close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and EMA trend filter
            # Volume filter: volume > 1.5x 20-period daily average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: 12h close > 1d EMA(50) for long, < for short
            ema_filter_long = close[i] > ema_50_aligned[i]
            ema_filter_short = close[i] < ema_50_aligned[i]
            
            # Long: price breaks above Donchian high + volume filter + EMA filter
            if close[i] > highest_high[i] and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + EMA filter
            elif close[i] < lowest_low[i] and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals