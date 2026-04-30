#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h Supertrend filter and volume confirmation
# Uses discrete sizing 0.20 to limit fee drag. Target: 60-150 total trades over 4 years (15-37/year).
# 4h Supertrend provides robust trend direction; 1h EMA(8/21) captures pullback entries.
# Volume spike ensures institutional participation. Session filter (08-20 UTC) reduces noise.
# Works in both bull and bear via 4h trend filter - only trades in direction of 4h trend.

name = "1h_EMA8_21_4hSupertrend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h EMA(8) and EMA(21)
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 4h Supertrend for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1_4h = high_4h[1:] - low_4h[1:]
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Supertrend
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + (multiplier * atr_4h)
    lower_band_4h = hl2_4h - (multiplier * atr_4h)
    
    supertrend_4h = np.zeros_like(close_4h)
    direction_4h = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend_4h[0] = upper_band_4h[0]
    direction_4h[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band_4h[i-1]:
            direction_4h[i] = 1
        elif close_4h[i] < lower_band_4h[i-1]:
            direction_4h[i] = -1
        else:
            direction_4h[i] = direction_4h[i-1]
            if direction_4h[i] == 1 and lower_band_4h[i] < lower_band_4h[i-1]:
                lower_band_4h[i] = lower_band_4h[i-1]
            if direction_4h[i] == -1 and upper_band_4h[i] > upper_band_4h[i-1]:
                upper_band_4h[i] = upper_band_4h[i-1]
        
        if direction_4h[i] == 1:
            supertrend_4h[i] = lower_band_4h[i]
        else:
            supertrend_4h[i] = upper_band_4h[i]
    
    # Align 4h Supertrend and direction to 1h
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR for stoploss (14-period) on 1h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or
            np.isnan(supertrend_4h_aligned[i]) or np.isnan(direction_4h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_8 = ema_8[i]
        curr_ema_21 = ema_21[i]
        curr_supertrend_4h = supertrend_4h_aligned[i]
        curr_direction_4h = direction_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with EMA crossover and 4h Supertrend filter
            if curr_volume_spike:
                # Bullish: EMA8 crosses above EMA21 + 4h uptrend
                if curr_ema_8 > curr_ema_21 and curr_direction_4h == 1:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish: EMA8 crosses below EMA21 + 4h downtrend
                elif curr_ema_8 < curr_ema_21 and curr_direction_4h == -1:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry OR Supertrend flip
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR EMA8 crosses below EMA21 OR Supertrend turns down
            if curr_low <= stop_loss or curr_ema_8 < curr_ema_21 or curr_direction_4h == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry OR Supertrend flip
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR EMA8 crosses above EMA21 OR Supertrend turns up
            if curr_high >= stop_loss or curr_ema_8 > curr_ema_21 or curr_direction_4h == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals