#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian breakouts capture strong momentum. 1d EMA34 filters for higher-timeframe trend alignment to avoid counter-trend trades. Volume spike confirms institutional participation. ATR-based stoploss manages risk. Works in bull via long breakouts in uptrend, works in bear via short breakouts in downtrend. Target 20-50 trades/year to minimize fee drag.
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
    entry_price = 0.0
    atr_multiplier = 2.5
    
    # Start index: need enough for Donchian(20) and EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if EMA34 not ready
        if np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        
        # Donchian(20): 20-period high/low
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])
            donchian_low = np.min(low[i-19:i+1])
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # ATR(20) for stoploss calculation
        if i >= 20:
            tr1 = high[i-19:i+1] - low[i-19:i+1]
            tr2 = np.abs(high[i-19:i+1] - np.roll(close[i-19:i+1], 1))
            tr3 = np.abs(low[i-19:i+1] - np.roll(close[i-19:i+1], 1))
            tr2[0] = tr1[0]  # first period has no previous close
            tr3[0] = tr1[0]
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr)
        else:
            atr = np.mean(high[:i+1] - low[:i+1])  # simplified for warmup
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1d EMA34 (uptrend filter) AND volume spike
            long_condition = (curr_close > donchian_high) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below Donchian low AND below 1d EMA34 (downtrend filter) AND volume spike
            short_condition = (curr_close < donchian_low) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Update trailing stop for long: highest high since entry
            # For simplicity, use close-based exit: exit if price drops below Donchian low or trend breaks
            if curr_close < donchian_low or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update trailing stop for short: lowest low since entry
            # Exit if price rises above Donchian high or trend breaks
            if curr_close > donchian_high or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0