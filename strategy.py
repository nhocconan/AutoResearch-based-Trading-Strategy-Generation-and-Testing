#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter + volume confirmation
    # Long: price > H3 + price > 4h EMA50 + volume > 1.5x 20-period average
    # Short: price < L3 + price < 4h EMA50 + volume > 1.5x 20-period average
    # Exit: opposite Camarilla breakout OR price crosses 4h EMA50
    # Using 1h timeframe with 4h/1d filters to reduce noise, targeting 15-30 trades/year.
    # Session filter: 08-20 UTC to avoid low-liquidity periods.
    # Discrete position sizing (0.20) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Get 1d data for Camarilla pivot calculation (using prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3) from prior 1d bar
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Prior day's range
        day_high = high_1d[i-1]
        day_low = low_1d[i-1]
        day_close = close_1d[i-1]
        day_range = day_high - day_low
        
        if day_range > 0:
            camarilla_h3[i] = day_close + day_range * 1.1 / 4
            camarilla_l3[i] = day_close - day_range * 1.1 / 4
        else:
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
    
    # Align Camarilla levels to 1h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 with min_periods
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])  # SMA50 as seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Align 4h EMA50 to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(close_4h), np.nan)
    for i in range(20, len(close_4h)):
        vol_ma_4h[i] = np.mean(vol_4h[i-20:i])
    volume_spike_4h = vol_4h > (1.5 * vol_ma_4h)
    
    # Align 4h volume spike to 1h
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla levels
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Trend filter from 4h EMA50
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and (volume_spike_4h_aligned[i] > 0.5)
        short_entry = short_breakout and bearish_trend and (volume_spike_4h_aligned[i] > 0.5)
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or (close[i] < ema_4h_aligned[i])
        short_exit = long_breakout or (close[i] > ema_4h_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_breakout_ema50_volume_v1"
timeframe = "1h"
leverage = 1.0