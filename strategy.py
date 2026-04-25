#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA50 Trend and Volume Spike + ATR Trailing Stop
Hypothesis: Daily Donchian channel (20-day high/low) acts as strong support/resistance.
Breakouts above 20-day high or below 20-day low with volume confirmation and 1w EMA50
trend filter capture strong momentum moves. Works in bull markets via long breakouts and
in bear markets via short breakdowns. ATR-based trailing stop manages risk. Target: 30-100
trades over 4 years (7-25/year) to minimize fee drag and improve test generalization.
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
    
    # Get 1d data for Donchian channel and 1w data for EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channel for 1d (based on previous day's OHLC)
    donch_high = np.full(len(df_1d), np.nan)
    donch_low = np.full(len(df_1d), np.nan)
    
    for i in range(20, len(df_1d)):
        donch_high[i] = np.max(df_1d['high'].iloc[i-20:i])
        donch_low[i] = np.min(df_1d['low'].iloc[i-20:i])
    
    # Align Donchian levels to 1d timeframe (no shift needed as we use previous day's values)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate 20-period volume MA for volume confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (1d)
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
    
    # Start index: need enough for EMA50_1w, Donchian, volume MA, ATR to propagate
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
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
        ema50_1w = ema_50_1w_aligned[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (strict filter)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: close above 20-day high with volume confirmation and 1w EMA50 uptrend
            long_breakout = (curr_close > donch_high) and volume_confirm and (curr_close > ema50_1w)
            # Short breakdown: close below 20-day low with volume confirmation and 1w EMA50 downtrend
            short_breakout = (curr_close < donch_low) and volume_confirm and (curr_close < ema50_1w)
            
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

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0