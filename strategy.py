#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
# Long when price breaks above 20-period high + volume > MA + close > 1d EMA200.
# Short when price breaks below 20-period low + volume > MA + close < 1d EMA200.
# Uses ATR-based stoploss to limit drawdown. Designed for low trade frequency (12-37/year)
# to minimize fee drag while capturing trends in both bull and bear markets.

name = "12h_donchian20_1d_ema200_vol_v1"
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
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA200 on 1d close
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian channels (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > vol_ma[i]
        trend_filter_long = close[i] > ema200_aligned[i]
        trend_filter_short = close[i] < ema200_aligned[i]
        
        if position == 1:  # long position
            # Exit: stoploss (2*ATR) or breakdown below Donchian low
            if close[i] < entry_price - 2.0 * atr[i] or close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss (2*ATR) or breakout above Donchian high
            if close[i] > entry_price + 2.0 * atr[i] or close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend filters
            if vol_filter:
                # Long breakout: price above Donchian high + uptrend
                if close[i] > high_max[i] and trend_filter_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price below Donchian low + downtrend
                elif close[i] < low_min[i] and trend_filter_short:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals