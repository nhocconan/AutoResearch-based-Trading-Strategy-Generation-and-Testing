#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In trending markets (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In ranging markets (ADX<20): fade extremes - short when Bull Power > 0.5*ATR, long when Bear Power < -0.5*ATR.
# Volume confirmation avoids low-liquidity false signals. Works in both bull and bear markets by adapting to regime.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "6h_ElderRay_1dADX25_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    # Calculate 1d +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate 1d ADX (14-period)
    period = 14
    atr_1d = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h ATR for Elder Ray thresholds
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 6h volume confirmation: >1.3x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.3 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough for all indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(ema_13_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Elder Ray calculations
        bull_power = high[i] - ema_13_6h[i]
        bear_power = low[i] - ema_13_6h[i]
        
        # Regime filters
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trending market logic: follow Elder Ray momentum
        if trending and vol_confirm:
            # Long when bull power positive and increasing
            long_entry = bull_power > 0 and bull_power > (bull_power if i==start_idx else bull_power_prev)
            # Short when bear power negative and decreasing (more negative)
            short_entry = bear_power < 0 and bear_power < (bear_power if i==start_idx else bear_power_prev)
            
            # Exit when Elder Ray diverges from price
            long_exit = bull_power < 0
            short_exit = bear_power > 0
            
        # Ranging market logic: fade Elder Ray extremes
        elif ranging and vol_confirm:
            # Short when bull power excessively positive
            short_entry = bull_power > 0.5 * atr_6h[i]
            # Long when bear power excessively negative
            long_entry = bear_power < -0.5 * atr_6h[i]
            
            # Exit when Elder Ray returns to neutral
            long_exit = bear_power >= -0.2 * atr_6h[i]
            short_exit = bull_power <= 0.2 * atr_6h[i]
            
        # Transition zone (20 <= ADX <= 25): no new entries, hold or exit
        else:
            long_entry = short_entry = False
            long_exit = position == 1
            short_exit = position == -1
        
        # Store previous values for next iteration
        bull_power_prev = bull_power
        bear_power_prev = bear_power
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals