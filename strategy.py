#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Weekly EMA50 Trend + Volume Spike
Hypothesis: Daily Donchian breakouts capture major market moves. Weekly EMA50 filter ensures we trade with the higher-timeframe trend, reducing false signals in ranging markets. Volume spike confirms institutional participation. Works in bull/bear via trend filter - only long in uptrend, short in downtrend. Target: 7-25 trades/year on 1d.
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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian calculation
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_1w_aligned[i]
        atr_val = atr[i]
        
        # Donchian channels (20-period)
        if i >= 20:
            highest_20 = np.max(high[i-19:i+1])
            lowest_20 = np.min(low[i-19:i+1])
        else:
            highest_20 = np.max(high[:i+1])
            lowest_20 = np.min(low[:i+1])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND uptrend
            long_condition = (curr_high > highest_20) and volume_spike and uptrend
            # Short: price breaks below Donchian low AND volume spike AND downtrend
            short_condition = (curr_low < lowest_20) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or trend reversal
            if curr_close <= entry_price - 2.0 * atr_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or trend reversal
            if curr_close >= entry_price + 2.0 * atr_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0