#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Triple_Confirmation_Strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d RSI(14) for overbought/oversold conditions
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 4h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume filter (above average volume)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema50_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_avg = vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume above average, 
            # daily trend up (price > EMA50), not overbought (RSI < 70)
            if (close[i] > upper_channel and volume[i] > vol_avg and 
                close[i] > ema_val and rsi_val < 70):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume above average,
            # daily trend down (price < EMA50), not oversold (RSI > 30)
            elif (close[i] < lower_channel and volume[i] > vol_avg and 
                  close[i] < ema_val and rsi_val > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR RSI overbought (>70)
            if close[i] < lower_channel or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR RSI oversold (<30)
            if close[i] > upper_channel or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines Donchian breakout with volume confirmation and daily trend/RSI filters.
# - Uses 4h Donchian channel (20) for breakout signals
# - Requires volume above 20-period average for confirmation
# - Uses 1d EMA(50) for trend direction filter
# - Uses 1d RSI(14) to avoid overextended entries (long only when RSI<70, short only when RSI>30)
# - Works in both bull and bear markets by following the daily trend
# - Volume filter reduces false breakouts from low-volume moves
# - Target: 80-150 total trades over 4 years (20-38/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return
# - Exit on opposite Donchian break or RSI extreme to lock in profits and avoid reversals