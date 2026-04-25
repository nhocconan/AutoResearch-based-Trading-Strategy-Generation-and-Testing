#!/usr/bin/env python3
"""
12h Donchian(20) Breakout with 1d EMA34 Trend and Volume Spike + ATR Stop
Hypothesis: 12h Donchian breakouts capture medium-term momentum. Filtered by 1d EMA34 trend for higher timeframe alignment and volume spike for confirmation. ATR-based stoploss manages risk. Designed for low trade frequency (12-37/year) to work in both bull and bear markets via trend following with strict entry conditions.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian channels (20-period)
    # We need to calculate Donchian on 12h data, then align to LTF
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian upper/lower on 12h close
    donch_high_12h = pd.Series(df_12h['close'].values).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(df_12h['close'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to LTF (12h primary timeframe)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate ATR for stoploss (using 12h data)
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    # Start index: need enough for all indicators
    start_idx = max(34, 20, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_12h[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > donch_high) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Donchian lower AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < donch_low) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: ATR-based stoploss OR price crosses below Donchian lower (breakdown) OR price crosses below EMA (trend change)
            stop_price = entry_price - (atr_stop_multiplier * atr_val)
            if (curr_close < stop_price) or (curr_close < donch_low) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: ATR-based stoploss OR price crosses above Donchian upper (breakout) OR price crosses above EMA (trend change)
            stop_price = entry_price + (atr_stop_multiplier * atr_val)
            if (curr_close > stop_price) or (curr_close > donch_high) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0