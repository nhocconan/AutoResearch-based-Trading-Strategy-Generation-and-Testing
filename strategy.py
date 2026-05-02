#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# ADX > 25 indicates trending market (use Elder Ray for trend continuation), ADX < 20 indicates ranging (fade extremes)
# Volume spike (1.5x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe
# Works in bull markets via Bull Power + ADX > 25, in bear markets via Bear Power + ADX > 25, and ranges via mean reversion at extremes

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    df_1d_high = pd.Series(df_1d['high'].values)
    df_1d_low = pd.Series(df_1d['low'].values)
    df_1d_close = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = abs(df_1d_high - df_1d_close.shift(1))
    tr3 = abs(df_1d_low - df_1d_close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d_high - df_1d_high.shift(1)
    down_move = df_1d_low.shift(1) - df_1d_low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_6h
    bear_power = low - ema_6h
    
    # Calculate volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume MA)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Trending market (ADX > 25): Elder Ray continuation
            if adx_1d_aligned[i] > 25:
                # Long: Bull Power > 0 (bullish momentum) + volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (bearish momentum) + volume spike
                elif bear_power[i] < 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX < 20): fade extremes at Elder Ray extremes
            elif adx_1d_aligned[i] < 20:
                # Long: Bear Power < -1 std dev (oversold) + volume spike
                # Short: Bull Power > +1 std dev (overbought) + volume spike
                # Use rolling std of Elder Ray for dynamic extremes
                if i >= 30:  # need history for std calculation
                    bull_std = pd.Series(bull_power[max(0, i-29):i+1]).std()
                    bear_std = pd.Series(bear_power[max(0, i-29):i+1]).std()
                    if not (np.isnan(bull_std) or np.isnan(bear_std)) and bull_std > 0 and bear_std > 0:
                        if bear_power[i] < -bull_std and volume_spike[i]:
                            signals[i] = 0.25
                            position = 1
                        elif bull_power[i] > bear_std and volume_spike[i]:
                            signals[i] = -0.25
                            position = -1
            else:
                # Transition zone (ADX 20-25): no new entries
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Elder Ray turns bearish OR ADX drops below 20 (trend weakening)
            if bull_power[i] <= 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Elder Ray turns bullish OR ADX drops below 20 (trend weakening)
            if bear_power[i] >= 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals