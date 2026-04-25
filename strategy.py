#!/usr/bin/env python3
"""
4h Volume Spike + EMA Trend Pullback v1
Hypothesis: In strong trends (BTC/ETH), price pulls back to the 21-period EMA on 4h during high-volume bars, offering high-probability entries. Volume > 2x 20-bar MA confirms institutional interest. Works in bull markets via long pullbacks to rising EMA and in bear markets via short pullbacks to falling EMA. Uses 1d EMA50 as higher-timeframe trend filter to avoid counter-trend trades. ATR-based trailing stop (2.0*ATR) controls risk. Targets 20-50 trades/year to minimize fee drag.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h EMA21 for pullback entries
    ema_21 = np.full(n, np.nan)
    close_series = pd.Series(close)
    ema_21_values = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21[:] = ema_21_values
    
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
    
    # Start index: need enough for EMA21, EMA50_1d, volume MA, ATR to propagate
    start_idx = max(21, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema21 = ema_21[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (strict filter)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long pullback: price near rising EMA21 with volume confirmation and 1d EMA50 uptrend
            long_pullback = (curr_close <= ema21 * 1.005) and (curr_close >= ema21 * 0.995) and volume_confirm and (ema50_1d > ema_50_1d_aligned[i-1])
            # Short pullback: price near falling EMA21 with volume confirmation and 1d EMA50 downtrend
            short_pullback = (curr_close <= ema21 * 1.005) and (curr_close >= ema21 * 0.995) and volume_confirm and (ema50_1d < ema_50_1d_aligned[i-1])
            
            if long_pullback:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_pullback:
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

name = "4h_Volume_Spike_EMA21_Pullback_1dEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0