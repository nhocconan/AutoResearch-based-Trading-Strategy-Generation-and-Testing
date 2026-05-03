#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA34 trend + volume confirmation
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND price > lips AND 1d EMA34 uptrend AND volume > 1.5x 20-bar average
# Short when jaws < teeth < lips AND price < lips AND 1d EMA34 downtrend AND volume > 1.5x 20-bar average
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR, short exit when price > lowest_low_since_entry + 2.0 * ATR
# Williams Alligator uses smoothed moving averages (SMMA) to identify trend alignment and avoid choppy markets.
# 1d EMA34 provides higher-timeframe bias filter. Volume confirms conviction.
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.25) to minimize fee drag.

name = "4h_WilliamsAlligator_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(source) < length:
        return np.full_like(source, np.nan, dtype=float)
    result = np.empty_like(source)
    result[:] = np.nan
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator (SMMA-based)
    # Jaws: 13-period SMMA of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    jaws = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 34) + 1  # volume MA(20) + EMA34(1d) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: jaws > teeth > lips AND price > lips AND 1d EMA34 uptrend AND volume spike
            if (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                close[i] > lips[i] and close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: jaws < teeth < lips AND price < lips AND 1d EMA34 downtrend AND volume spike
            elif (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                  close[i] < lips[i] and close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals