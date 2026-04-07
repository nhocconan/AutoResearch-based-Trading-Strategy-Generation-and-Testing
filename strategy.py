#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA50) and volume confirmation
# Uses Donchian channel breakouts for trend following, confirmed by 12h EMA trend direction
# and volume above 20-period average. Includes ATR-based stoploss to limit drawdowns.
# Designed for moderate trade frequency (target: 20-50 trades/year) to balance signal quality
# and fee efficiency. Works in bull markets via breakouts and in bear markets via
# short breakdowns with trend filter alignment.

name = "4h_donchian20_12h_ema_volume_v1"
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
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above previous high
        short_breakout = close[i] < lowest_low[i-1]   # Break below previous low
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Exit conditions: reverse signal or stoploss
        if position == 1:  # Long position
            # Exit on reverse breakout or stoploss (2*ATR below entry)
            if short_breakout or close[i] <= lowest_low[i-1] + 2 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse breakout or stoploss (2*ATR above entry)
            if long_breakout or close[i] >= highest_high[i-1] - 2 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish breakout with uptrend and volume confirmation
            if long_breakout and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.30
            # Short entry: bearish breakout with downtrend and volume confirmation
            elif short_breakout and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals