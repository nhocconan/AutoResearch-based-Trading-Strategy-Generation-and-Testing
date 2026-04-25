#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture multi-day momentum in BTC/ETH. The 1d EMA34 filter ensures alignment with higher timeframe trend, working in both bull/bear markets. Volume confirmation filters false breakouts. 12h timeframe targets 12-37 trades/year to minimize fee drag while capturing significant moves. ATR-based stoploss manages risk.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr = np.zeros(n)  # ATR for stoploss
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Start index: need enough for Donchian(20) and ATR(14) warmup
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_val = atr[i]
        
        # Donchian(20) channels: highest high and lowest low of last 20 periods
        if i >= 20:
            highest_high = np.max(high[i-19:i+1])
            lowest_low = np.min(low[i-19:i+1])
        else:
            highest_high = np.max(high[:i+1])
            lowest_low = np.min(low[:i+1])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above Donchian upper channel AND above daily EMA34 (uptrend filter)
            long_condition = (curr_close > highest_high) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below Donchian lower channel AND below daily EMA34 (downtrend filter)
            short_condition = (curr_close < lowest_low) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Trail stop: exit if price drops 2*ATR from highest high since entry
            # Simplified: exit if price closes below Donchian lower channel or trend breaks
            if curr_close <= lowest_low or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Trail stop: exit if price rises 2*ATR from lowest low since entry
            # Simplified: exit if price closes above Donchian upper channel or trend breaks
            if curr_close >= highest_high or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0