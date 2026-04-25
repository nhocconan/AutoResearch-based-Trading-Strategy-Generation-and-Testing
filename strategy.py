#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_ATR_Volume
Hypothesis: On 12h timeframe, Donchian channel (20-period) breakouts with 1d EMA50 trend filter, volume confirmation (>1.5x 20-bar avg), and ATR-based trailing stop (2.5x ATR) captures medium-term trends while minimizing overtrading. The 12h timeframe reduces noise, trend filter ensures directional alignment in bull/bear markets, volume confirms breakout validity, and ATR stop manages risk. Discrete sizing (0.25) limits fee churn. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 12h
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for volatility and stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        ema_val = ema_50_aligned[i]
        upper_donchian = high_ma[i]
        lower_donchian = low_ma[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry: Donchian breakout with trend and volume
            # Long: price breaks above upper Donchian with uptrend (close > EMA50) and volume
            long_signal = (high_val > upper_donchian) and (close_val > ema_val) and volume_confirm
            # Short: price breaks below lower Donchian with downtrend (close < EMA50) and volume
            short_signal = (low_val < lower_donchian) and (close_val < ema_val) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                highest_since_entry = high_val  # initialize trailing stop high
            elif short_signal:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low_val   # initialize trailing stop low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest high
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest low
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_ATR_Volume"
timeframe = "12h"
leverage = 1.0