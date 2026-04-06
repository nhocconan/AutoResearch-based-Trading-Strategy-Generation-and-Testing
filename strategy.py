#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA trend filter + volume confirmation
# Enter long when price breaks above Donchian upper band in uptrend (12h EMA > prior EMA)
# Enter short when price breaks below Donchian lower band in downtrend (12h EMA < prior EMA)
# Volume > 1.5x 20-period average for confirmation
# Uses discrete position sizes (0.30) to minimize churn, ATR-based stoploss
# Targets 80-180 total trades over 4 years (20-45/year) by requiring trend alignment

name = "4h_donchian_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_prev = np.roll(ema_12h, 1)
    ema_12h_prev[0] = ema_12h[0]
    ema_12h_rising = ema_12h > ema_12h_prev  # Uptrend
    ema_12h_falling = ema_12h < ema_12h_prev  # Downtrend
    
    # Align 12h EMA trends to 4h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR for stoploss (2-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(volume_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Maintain current position
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: price drops 2*ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price falls below Donchian lower band
            elif close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: price rises 2*ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price rises above Donchian upper band
            elif close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: breakout with volume and trend alignment
            if ema_rising_aligned[i]:  # Uptrend
                if close[i] > highest_high[i] and volume[i] > volume_threshold[i]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = close[i]
            elif ema_falling_aligned[i]:  # Downtrend
                if close[i] < lowest_low[i] and volume[i] > volume_threshold[i]:
                    signals[i] = -0.30
                    position = -1
                    entry_price = close[i]
    
    return signals