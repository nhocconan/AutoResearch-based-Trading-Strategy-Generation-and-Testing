#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation (>1.5x 20-bar avg), and ATR(14) stoploss.
# Uses discrete position sizing (±0.25) to manage fee drag. Target: 80-160 total trades over 4 years (20-40/year).
# Works in bull markets via breakout continuation and in bear markets via volatility expansion capture after squeezes.
# Entry: price breaks Donchian channel + trend alignment + volume spike.
# Exit: ATR-based stoploss (2.0 * ATR) or time-based exit (10 bars) to prevent whipsaws.

name = "4h_Donchian20_1dEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d_vals).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian Channel (20) on 4h
    dc_period = 20
    upper_channel = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_channel = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # ATR(14) for stoploss and volatility filter
    atr_period = 14
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
    tr3 = abs(pd.Series(low).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    start_idx = max(dc_period, atr_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian channel, close > 1d EMA34, volume spike
            if (curr_high > curr_upper and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            # Short: price breaks below lower Donchian channel, close < 1d EMA34, volume spike
            elif (curr_low < curr_lower and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
        
        elif position == 1:  # Long position
            bars_since_entry += 1
            # Exit conditions: ATR stoploss, time-based exit, or mean reversion to middle
            stop_loss = entry_price - (2.0 * curr_atr)
            if (curr_low <= stop_loss or  # ATR stoploss hit
                bars_since_entry >= 10):   # time-based exit (max 10 bars)
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit conditions: ATR stoploss, time-based exit, or mean reversion to middle
            stop_loss = entry_price + (2.0 * curr_atr)
            if (curr_high >= stop_loss or  # ATR stoploss hit
                bars_since_entry >= 10):   # time-based exit (max 10 bars)
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals