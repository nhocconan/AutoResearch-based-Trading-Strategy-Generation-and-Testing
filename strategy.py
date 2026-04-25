#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Daily Donchian channel breakouts capture significant multi-day moves. 
Weekly EMA50 filters for higher-timeframe trend alignment, reducing false breakouts.
Volume spike confirms institutional participation. ATR-based stoploss manages risk.
Works in both bull and bear markets by trend-filtering breakouts and using volatility-based stops.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss and volume averaging
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume average for volume spike detection
    volume_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need enough for Donchian calculation (20) + EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_ma = volume_ma[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period EMA
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian channel AND above weekly EMA50 (uptrend filter)
            long_condition = (curr_close > upper_channel) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below lower Donchian channel AND below weekly EMA50 (downtrend filter)
            short_condition = (curr_close < lower_channel) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = atr_val
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = atr_val
        elif position == 1:
            # Long position management
            # Stoploss: 2.5 * ATR below entry
            stop_loss = entry_price - 2.5 * atr_at_entry
            # Exit conditions: stoploss hit OR price returns to midpoint of channel
            midpoint = (upper_channel + lower_channel) / 2.0
            if curr_low <= stop_loss or curr_close <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * atr_at_entry
            # Exit conditions: stoploss hit OR price returns to midpoint of channel
            midpoint = (upper_channel + lower_channel) / 2.0
            if curr_high >= stop_loss or curr_close >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0