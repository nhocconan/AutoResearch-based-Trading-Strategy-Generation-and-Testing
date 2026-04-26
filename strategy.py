#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1wTrend_VolumeConfirmation_ATRStop
Hypothesis: On 4h timeframe, Donchian(20) breakouts with 1-week EMA50 trend filter and volume confirmation (>1.5x 96-bar avg) capture strong momentum moves. Uses weekly trend to filter direction and avoid counter-trend trades. ATR-based stoploss limits downside in bear markets. Targets 20-40 trades/year to minimize fee drag while maintaining edge via trend filter and volume confirmation. Works in both bull (breakouts with trend) and bear (short breakdowns against weekly downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (96-period = 16 days on 4h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(100, 60, 96, 14)  # Donchian, 1w lookback, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_1w_val = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        atr_val = atr[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        highest_20 = np.max(high[lookback_start:i+1])
        lowest_20 = np.min(low[lookback_start:i+1])
        
        # Volume confirmation: current volume > 1.5x 96-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper band with uptrend (close > EMA50_1w) and volume confirmation
            long_signal = (high_val > highest_20) and (close_val > ema_50_1w_val) and volume_confirmed
            # Short: price breaks below Donchian lower band with downtrend (close < EMA50_1w) and volume confirmation
            short_signal = (low_val < lowest_20) and (close_val < ema_50_1w_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit conditions:
            # 1. ATR trailing stop: price drops 2.5*ATR from highest since entry
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # 2. Trend reversal: close crosses below EMA50_1w
            elif close_val < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # 3. Opposite Donchian breakout: price breaks below Donchian lower band
            elif low_val < lowest_20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit conditions:
            # 1. ATR trailing stop: price rises 2.5*ATR from lowest since entry
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # 2. Trend reversal: close crosses above EMA50_1w
            elif close_val > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # 3. Opposite Donchian breakout: price breaks above Donchian upper band
            elif high_val > highest_20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_1wTrend_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0