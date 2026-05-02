#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 1d ADX25 regime filter + volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# 1d ADX > 25 ensures we only trade in trending markets (avoids chop)
# Volume spike (1.8x 20-period average) confirms momentum
# Discrete sizing 0.25 targets ~50-150 trades over 4 years (12-38/year)
# Works in bull/bear via trend-following with regime filter to avoid false signals

name = "6h_ElderRay_1dADX25_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX25 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate ADX(25) on 1d for regime filter
    # ADX calculation: +DM, -DM, TR, then smoothed averages
    df_1d_copy = df_1d.copy()
    df_1d_copy['high'] = df_1d['high'].values
    df_1d_copy['low'] = df_1d['low'].values
    df_1d_copy['close'] = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_copy['high'] - df_1d_copy['low']
    tr2 = abs(df_1d_copy['high'] - df_1d_copy['close'].shift(1))
    tr3 = abs(df_1d_copy['low'] - df_1d_copy['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=25, adjust=False, min_periods=25).mean()
    
    # Directional Movement
    up_move = df_1d_copy['high'] - df_1d_copy['high'].shift(1)
    down_move = df_1d_copy['low'].shift(1) - df_1d_copy['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=25, adjust=False, min_periods=25).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr.values
    minus_di = 100 * minus_dm_smooth / atr.values
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=25, adjust=False, min_periods=25).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray on 6h: EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Elder Ray signals: rising power indicates strength
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    # Handle first element
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    # Volume confirmation (1.8x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and ADX calculations)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_rising[i]) or 
            np.isnan(bear_power_rising[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising + ADX > 25 (trending) + volume spike
            if bull_power[i] > 0 and bull_power_rising[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 and rising + ADX > 25 (trending) + volume spike
            elif bear_power[i] > 0 and bear_power_rising[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power becomes negative or stops rising or ADX weakens
            if bull_power[i] <= 0 or not bull_power_rising[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power becomes negative or stops rising or ADX weakens
            if bear_power[i] <= 0 or not bear_power_rising[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals