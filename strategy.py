#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation
# Long when price breaks above upper Donchian channel + price > 1d EMA50 + volume > 1.5x average
# Short when price breaks below lower Donchian channel + price < 1d EMA50 + volume > 1.5x average
# Uses tight volatility-based stoploss: exit when price moves 2.5*ATR against position
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets via breakouts, in bear markets via short breakdowns

name = "12h_donchian_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: exit if price drops 2.5*ATR below entry
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price closes below midpoint of Donchian channel
            elif close[i] < (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: exit if price rises 2.5*ATR above entry
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price closes above midpoint of Donchian channel
            elif close[i] > (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout/breakdown entries
            # Long: price breaks above upper Donchian + above 1d EMA50 + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower Donchian + below 1d EMA50 + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals