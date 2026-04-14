#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation
# Long when price breaks above 1h Keltner upper band, 4h trend is bullish (price > 4h EMA50), and volume > 1.5x 20-period average
# Short when price breaks below 1h Keltner lower band, 4h trend is bearish (price < 4h EMA50), and volume > 1.5x 20-period average
# Exit when price crosses 1h Keltner middle line
# Uses 4h EMA50 as trend filter to avoid counter-trend trades
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag
# Session filter: 08-20 UTC to reduce noise trades

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1h Keltner Channel (20-period, ATR multiplier 2.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    keltner_middle = ema_20
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1h volume average (20-period)
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_4h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_4h, keltner_lower)
    keltner_middle_aligned = align_htf_to_ltf(prices, df_4h, keltner_middle)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # for 20-period calculations and EMA50
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long setup: break above Keltner upper with volume confirmation and 4h bullish trend
            if (price > keltner_upper_aligned[i] and 
                vol_current > 1.5 * vol_ma_1h_aligned[i] and  # Volume confirmation
                price > ema_50_4h_aligned[i]):               # 4h trend filter - bullish
                position = 1
                signals[i] = position_size
            # Short setup: break below Keltner lower with volume confirmation and 4h bearish trend
            elif (price < keltner_lower_aligned[i] and 
                  vol_current > 1.5 * vol_ma_1h_aligned[i] and  # Volume confirmation
                  price < ema_50_4h_aligned[i]):               # 4h trend filter - bearish
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Keltner middle
            if price < keltner_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Keltner middle
            if price > keltner_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Keltner_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0