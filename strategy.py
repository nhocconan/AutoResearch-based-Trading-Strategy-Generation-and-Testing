#!/usr/bin/env python3
"""
1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation + ATR stoploss
Hypothesis: Daily Donchian breakouts with weekly EMA trend filter capture momentum in both bull and bear markets.
Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. Targets 7-25 trades/year
to minimize fee drag while maintaining edge. Works in bull via breakout continuation, in bear via
mean-reversion from extreme levels when aligned with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: highest high over past 20 days
    donchian_high = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
    
    # Lower channel: lowest low over past 20 days
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(low_1d)):
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        ema_50_1w = np.full(n, np.nan)  # Default to no trend if no weekly data
    else:
        # Calculate 50-period EMA on weekly close
        close_1w = df_1w['close'].values
        ema_50_1w_raw = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w = align_htf_to_ltf(prices, df_1w, ema_50_1w_raw)
    
    # Calculate ATR(14) for stoploss on 1d data
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, ATR, and volume MA to propagate
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr[i]) or
            (len(df_1w) >= 2 and np.isnan(ema_50_1w[i]))):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_channel = donchian_high_aligned[i]
        lower_channel = donchian_low_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        ema50_1w = ema_50_1w[i] if len(df_1w) >= 2 else np.nan
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma
        
        # Trend filter: only trade in direction of weekly EMA50
        if len(df_1w) >= 2:
            uptrend = curr_close > ema50_1w
            downtrend = curr_close < ema50_1w
        else:
            uptrend = True  # No weekly data, allow both directions
            downtrend = True
        
        if position == 0:
            # Long: price breaks above upper channel AND uptrend AND volume spike
            long_condition = (curr_close > upper_channel) and uptrend and volume_spike
            # Short: price breaks below lower channel AND downtrend AND volume spike
            short_condition = (curr_close < lower_channel) and downtrend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below lower channel (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above upper channel (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0