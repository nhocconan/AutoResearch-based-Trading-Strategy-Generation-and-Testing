#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian breakouts capture strong momentum moves. EMA34 on 1d filters for higher timeframe trend alignment.
Volume spike confirms breakout strength. ATR-based stoploss limits downside. Designed for BTC/ETH with 75-150 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull (breakouts) and bear (breakdowns) via symmetric logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter (4h)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20), EMA34, ATR, volume MA
    start_idx = max(34, 20, 14)  # 34 for EMA34, 20 for Donchian/volume MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Donchian(20) channels
        if i >= 20:
            donchian_high = np.max(high[i-20:i])  # past 20 bars, not including current
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = np.max(high[:i]) if i > 0 else curr_high
            donchian_low = np.min(low[:i]) if i > 0 else curr_low
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above Donchian high + volume + price > 1d EMA34
            long_breakout = (curr_close > donchian_high) and volume_confirm and (curr_close > ema_34_val)
            # Short breakdown: price breaks below Donchian low + volume + price < 1d EMA34
            short_breakdown = (curr_close < donchian_low) and volume_confirm and (curr_close < ema_34_val)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakdown:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Long exit: stoploss hit OR price closes below Donchian low
            stoploss_level = entry_price - 2.5 * atr_val
            if curr_close <= stoploss_level or curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: stoploss hit OR price closes above Donchian high
            stoploss_level = entry_price + 2.5 * atr_val
            if curr_close >= stoploss_level or curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0