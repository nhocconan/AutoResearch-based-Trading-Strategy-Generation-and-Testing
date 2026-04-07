#!/usr/bin/env python3
"""
4h_atr_breakout_12h_trend_volume_v1
Hypothesis: On 4h timeframe, use ATR-based breakouts from prior 12h high/low with 12h EMA trend filter and volume confirmation. Enter long when price breaks above prior 12h high + 0.5*ATR with 12h EMA20 > EMA50 and volume > 1.5x average; enter short when price breaks below prior 12h low - 0.5*ATR with 12h EMA20 < EMA50 and volume > 1.5x average. Exit on opposite breakout or EMA reversal. This combines volatility breakouts with trend filtering to capture momentum while avoiding whipsaws. Targets 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA20 and EMA50 for 4h trend filter
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Get 12h data for breakout levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h high/low for breakout levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA20/EMA50 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 12h data to 4h timeframe (shifted by 1 bar for look-ahead prevention)
    high_12h_4h = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_4h = align_htf_to_ltf(prices, df_12h, low_12h)
    ema20_12h_4h = align_htf_to_ltf(prices, df_12h, ema20_12h)
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation (24-period average on 4h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(high_12h_4h[i]) or np.isnan(low_12h_4h[i]) or
            np.isnan(ema20_12h_4h[i]) or np.isnan(ema50_12h_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price breaks below prior 12h low (reverse breakout)
            if close[i] < low_12h_4h[i]:
                exit_long = True
            # Exit if 12h EMA20 crosses below EMA50 (trend reversal)
            elif ema20_12h_4h[i] < ema50_12h_4h[i] and ema20_12h_4h[i-1] >= ema50_12h_4h[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price breaks above prior 12h high (reverse breakout)
            if close[i] > high_12h_4h[i]:
                exit_short = True
            # Exit if 12h EMA20 crosses above EMA50 (trend reversal)
            elif ema20_12h_4h[i] > ema50_12h_4h[i] and ema20_12h_4h[i-1] <= ema50_12h_4h[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above prior 12h high + 0.5*ATR with 12h EMA20 > EMA50 and volume confirmation
            long_entry = False
            breakout_level = high_12h_4h[i] + 0.5 * atr[i]
            if (close[i] > breakout_level and close[i-1] <= breakout_level and
                ema20_12h_4h[i] > ema50_12h_4h[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below prior 12h low - 0.5*ATR with 12h EMA20 < EMA50 and volume confirmation
            short_entry = False
            breakdown_level = low_12h_4h[i] - 0.5 * atr[i]
            if (close[i] < breakdown_level and close[i-1] >= breakdown_level and
                ema20_12h_4h[i] < ema50_12h_4h[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals