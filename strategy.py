#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + 1d EMA34 Trend + ATR Trailing Stop
Hypothesis: Donchian channel breakouts capture institutional order flow with volume confirmation.
1d EMA34 filter ensures alignment with higher timeframe trend. Works in bull markets via long breakouts
and bear markets via short breakdowns. ATR-based trailing stop manages risk. Target: 75-200 trades/year.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period Donchian channels (4h)
    donch_high_20 = np.full(n, np.nan)
    donch_low_20 = np.full(n, np.nan)
    for i in range(20, n):
        donch_high_20[i] = np.max(high[i-19:i+1])
        donch_low_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (4h)
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Donchian, volume MA, ATR, and EMA34_1d to propagate
    start_idx = max(20, 20, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_20[i]) or 
            np.isnan(donch_low_20[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high = donch_high_20[i]
        donch_low = donch_low_20[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        ema34_1d = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average (balanced filter)
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Long breakout: close above Donchian high with volume confirmation and 1d EMA34 uptrend
            long_breakout = (curr_close > donch_high) and volume_confirm and (curr_close > ema34_1d)
            # Short breakdown: close below Donchian low with volume confirmation and 1d EMA34 downtrend
            short_breakout = (curr_close < donch_low) and volume_confirm and (curr_close < ema34_1d)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            atr_stop = max(atr_stop, curr_high - 2.0 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            atr_stop = min(atr_stop, curr_low + 2.0 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_1dEMA34_Trend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0