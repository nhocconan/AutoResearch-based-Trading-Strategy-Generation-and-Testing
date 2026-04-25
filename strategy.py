#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA200 Trend + Volume Spike + ATR Stoploss
Hypothesis: On 12h timeframe, Donchian channel breakouts (20-bar) capture strong momentum.
Filtered by 1-week EMA200 trend (bull/bear regime) and volume confirmation (>2x 20-bar avg volume).
ATR-based trailing stop (1.5x ATR) manages risk. Works in bull markets via long breakouts
and bear markets via short breakdowns. Low-frequency design targets 50-150 trades over 4 years
to minimize fee drag. Uses 1w and 1d HTF for trend and pivot context.
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
    
    # Get HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 1d Donchian(20) for structure (optional filter)
    donch_high_20_1d = np.full(len(df_1d), np.nan)
    donch_low_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        donch_high_20_1d[i] = np.max(df_1d['high'].iloc[i-19:i+1])
        donch_low_20_1d[i] = np.min(df_1d['low'].iloc[i-19:i+1])
    donch_high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20_1d)
    donch_low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20_1d)
    
    # 12h Donchian(20) breakout levels
    donch_high_20 = np.full(n, np.nan)
    donch_low_20 = np.full(n, np.nan)
    for i in range(20, n):
        donch_high_20[i] = np.max(high[i-19:i+1])
        donch_low_20[i] = np.min(low[i-19:i+1])
    
    # 12h volume MA(20) for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # 12h ATR(14) for stoploss
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
    
    # Start index: need enough for all indicators
    start_idx = max(200, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(donch_high_20[i]) or 
            np.isnan(donch_low_20[i]) or 
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
        ema200_1w = ema_200_1w_aligned[i]
        donch_high = donch_high_20[i]
        donch_low = donch_low_20[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: close above 12h Donchian HIGH with volume confirmation and 1w EMA200 uptrend
            long_breakout = (curr_close > donch_high) and volume_confirm and (curr_close > ema200_1w)
            # Short breakdown: close below 12h Donchian LOW with volume confirmation and 1w EMA200 downtrend
            short_breakout = (curr_close < donch_low) and volume_confirm and (curr_close < ema200_1w)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 1.5 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 1.5 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 1.5*ATR
            atr_stop = max(atr_stop, curr_high - 1.5 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 1.5*ATR
            atr_stop = min(atr_stop, curr_low + 1.5 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA200_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0