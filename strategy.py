#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d EMA200 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian(20) breakouts capture momentum bursts, filtered by 1d EMA200 trend (long in uptrend, short in downtrend) to avoid counter-trend whipsaws. Volume spike confirms institutional participation. ATR-based stoploss manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 20-40 trades/year (80-160 over 4 years).
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate ATR for volatility (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian (20) and EMA200 alignment
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_200_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_200_aligned[i]
        atr_value = atr[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above upper channel or below lower channel
        bullish_breakout = curr_close > upper_channel
        bearish_breakout = curr_close < lower_channel
        
        # Stoploss conditions: ATR-based trailing stop
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Long: stop if price drops below entry - 2.0 * ATR or bearish breakout of lower channel
                if curr_close < entry_price - 2.0 * atr_value or bearish_breakout:
                    exit_signal = True
                    
            elif position == -1:
                # Short: stop if price rises above entry + 2.0 * ATR or bullish breakout of upper channel
                if curr_close > entry_price + 2.0 * atr_value or bullish_breakout:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above upper channel AND price above 1d EMA200
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below lower channel AND price below 1d EMA200
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA200_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0