#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 12h EMA trend filter and volume confirmation.
# Camarilla levels identify key support/resistance where reversals occur.
# EMA(50) on 12h determines trend direction (only trade in trend direction).
# Volume > 1.5x average confirms participation. Works in both bull and bear markets by
# taking reversals in the direction of the higher timeframe trend.
# Target: 25-40 trades/year with disciplined entries to minimize fee drag.

name = "4h_Camarilla_Reversal_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA(50) on 12h close
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 49) / 51
    
    # Calculate 20-period average volume on 12h
    vol_avg_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_avg_20_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Calculate Camarilla levels on 4h data (using previous bar's OHLC)
    camarilla_H4 = np.full(n, np.nan)  # Resistance level
    camarilla_L4 = np.full(n, np.nan)  # Support level
    for i in range(1, n):
        # Use previous bar's OHLC to calculate today's levels
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_val = ph - pl
        camarilla_H4[i] = pc + (range_val * 1.1 / 2)  # H4 level
        camarilla_L4[i] = pc - (range_val * 1.1 / 2)  # L4 level
    
    # Align all indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_12h_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 1.5 * vol_avg_20_12h_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla level touch + EMA trend + volume
            # Long when price touches L4 support in uptrend (price > EMA)
            long_condition = (low[i] <= camarilla_L4[i] * 1.001) and \
                             (close[i] > ema_50_12h_aligned[i]) and vol_filter
            # Short when price touches H4 resistance in downtrend (price < EMA)
            short_condition = (high[i] >= camarilla_H4[i] * 0.999) and \
                              (close[i] < ema_50_12h_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches H4 resistance or trend changes
            if (high[i] >= camarilla_H4[i] * 0.999) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches L4 support or trend changes
            if (low[i] <= camarilla_L4[i] * 1.001) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals