#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume-weighted trend filter
# Uses Donchian channel breakouts confirmed by 1d VWAP trend and volume confirmation.
# Designed for 25-50 trades/year to minimize fee drag. Works in bull markets via breakouts
# and in bear markets via short breakdowns with VWAP trend alignment. Includes ATR-based
# stoploss to limit drawdowns during volatile periods.

name = "4h_donchian20_1d_vwap_volume_v1"
timeframe = "4h"
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
    
    # 1d VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP
    vwap_num = (high_1d + low_1d + close_1d) * volume_1d
    vwap_den = volume_1d
    vwap_cumsum = np.cumsum(vwap_num)
    vol_cumsum = np.cumsum(vwap_den)
    vwap_1d = vwap_cumsum / vol_cumsum
    vwap_1d = np.where(vwap_den == 0, np.nan, vwap_1d)
    vwap_1d = pd.Series(vwap_1d).ffill().bfill().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vwap_1d[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above previous high
        short_breakout = close[i] < lowest_low[i-1]   # Break below previous low
        
        # Trend filter from 1d VWAP
        uptrend = close[i] > vwap_1d[i]
        downtrend = close[i] < vwap_1d[i]
        
        # Exit conditions: reverse signal or stoploss
        if position == 1:  # Long position
            # Exit on reverse breakout or stoploss (2*ATR below entry)
            if short_breakout or close[i] <= lowest_low[i-1] + 2 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse breakout or stoploss (2*ATR above entry)
            if long_breakout or close[i] >= highest_high[i-1] - 2 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish breakout with uptrend and volume confirmation
            if long_breakout and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish breakout with downtrend and volume confirmation
            elif short_breakout and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals