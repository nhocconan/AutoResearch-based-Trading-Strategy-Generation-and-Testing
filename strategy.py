#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume confirmation.
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In trending regimes (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In ranging regimes (ADX<=25): fade extremes - short when Bull Power > 0.5*ATR, long when Bear Power < -0.5*ATR.
# Volume confirmation avoids low-liquidity false signals. Target: 50-150 total trades over 4 years = 12-37/year.
# Size: 0.25. Works in both bull and bear markets by adapting to regime.

name = "6h_ElderRay_1dADX25_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX regime filter and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h ATR(14) for Elder Ray scaling and volume MA
    # True Range for 6h
    tr1_6h = np.abs(high[1:] - low[:-1])
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(np.maximum(tr1_6h, tr2_6h), tr3_6h)
    atr_14_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(adx_14_aligned[i]) or
            np.isnan(atr_14_6h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = high[i] - ema_13_1d_aligned[i]
        bear_power = low[i] - ema_13_1d_aligned[i]
        
        # Regime filter: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_14_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if is_trending:
            # Trending regime: follow Elder Ray momentum
            long_entry = bull_power > 0 and bull_power > bull_power - (bull_power - (high[i-1] - ema_13_1d_aligned[i-1]) if i>0 and not np.isnan(high[i-1]) else 0) and vol_confirm
            short_entry = bear_power < 0 and bear_power < bear_power - (bear_power - (low[i-1] - ema_13_1d_aligned[i-1]) if i>0 and not np.isnan(low[i-1]) else 0) and vol_confirm
            
            # Simplified momentum check: rising bull power or falling bear power
            if i > 0:
                prev_bull_power = high[i-1] - ema_13_1d_aligned[i-1] if not np.isnan(ema_13_1d_aligned[i-1]) else 0
                prev_bear_power = low[i-1] - ema_13_1d_aligned[i-1] if not np.isnan(ema_13_1d_aligned[i-1]) else 0
                long_entry = bull_power > 0 and bull_power > prev_bull_power and vol_confirm
                short_entry = bear_power < 0 and bear_power < prev_bear_power and vol_confirm
            else:
                long_entry = bull_power > 0 and vol_confirm
                short_entry = bear_power < 0 and vol_confirm
        else:
            # Ranging regime: fade Elder Ray extremes
            atr_threshold = 0.5 * atr_14_6h[i]
            long_entry = bear_power < -atr_threshold and vol_confirm
            short_entry = bull_power > atr_threshold and vol_confirm
        
        # Exit conditions: opposite signal or power crosses zero
        long_exit = bull_power < 0
        short_exit = bear_power > 0
        
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