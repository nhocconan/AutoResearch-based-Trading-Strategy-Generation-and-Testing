#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ADX regime filter
# - Camarilla pivot levels from 1d: L3/H3 as breakout levels
# - 1d volume confirmation: current volume > 1.5x 20-period average to avoid false breakouts
# - ADX regime filter: only trade when ADX(14) > 25 (trending market)
# - Exit: Camarilla opposite pivot touch or ADX drops below 20
# - Target: 12-37 trades/year on 12h (50-150 total over 4 years) to avoid fee drag
# - Position size: 0.25 (25% of capital) for balanced risk/return

name = "12h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    L3 = close_1d - (range_1d * 1.1 / 4)
    H3 = close_1d + (range_1d * 1.1 / 4)
    L4 = close_1d - (range_1d * 1.1 / 2)
    H4 = close_1d + (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute 12h ADX for regime filter
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # True Range
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = -np.diff(low_12h, prepend=low_12h[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_period = 14
    tr_smoothed = wilders_smoothing(tr, atr_period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, atr_period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, atr_period)
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, atr_period)
    
    # Pre-compute 12h volume average (20-period)
    volume_12h = prices['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx[i]) or
            np.isnan(vol_ma_20_12h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirm = volume_12h[i] > 1.5 * vol_ma_20_12h[i]
        
        # ADX regime filter: only trade when trending (ADX > 25)
        trending_regime = adx[i] > 25
        
        # Camarilla breakout conditions
        long_breakout = close_12h[i] > H3_aligned[i]   # Break above H3
        short_breakout = close_12h[i] < L3_aligned[i]  # Break below L3
        
        # Exit conditions: Camarilla opposite pivot touch or regime change
        exit_long = close_12h[i] < L3_aligned[i]   # Price touches L3
        exit_short = close_12h[i] > H3_aligned[i]  # Price touches H3
        regime_exit = adx[i] < 20  # ADX drops below 20 (ranging market)
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Camarilla breakout up AND volume confirmation AND trending regime
            if long_breakout and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short conditions: Camarilla breakout down AND volume confirmation AND trending regime
            elif short_breakout and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: opposite Camarilla touch OR regime change OR volume exhaustion
            vol_exhaustion = volume_12h[i] < vol_ma_20_12h[i]
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short) or regime_exit or vol_exhaustion
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals