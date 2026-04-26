#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian_20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts only when aligned with 1d ADX regime (ADX>25 = trend, ADX<20 = range) and confirmed by volume spike. In trend regime (ADX>25), breakout in direction of 1d EMA50 trend. In range regime (ADX<20), fade at Donchian bands. Volume confirmation avoids false breakouts. This adapts to bull/bear markets via regime detection. Discrete sizing (0.25) limits fee drift. Target: 50-120 total trades over 4 years (12-30/year) by requiring regime alignment, breakout, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for regime detection
    # ADX requires +DI, -DI, TR
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # +DI and -DI
    up_move = high_1d - high_1d.shift(1)
    down_move = low_1d.shift(1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA50 for trend direction
    ema_50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Donchian(20) bands
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA50, 20 for Donchian/volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Regime conditions
        adx = adx_1d_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        close_price = close[i]
        
        # Trend regime: ADX > 25
        # Range regime: ADX < 20
        # Transition zone: 20 <= ADX <= 25 (no new entries, hold existing)
        
        if adx > 25:  # Trend regime
            # Breakout in direction of 1d EMA50 trend
            breakout_above = close_price > donchian_high[i]
            breakout_below = close_price < donchian_low[i]
            
            if breakout_above and volume_spike and close_price > ema_50:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif breakout_below and volume_spike and close_price < ema_50:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
                    
        elif adx < 20:  # Range regime
            # Fade at Donchian bands (mean reversion)
            if close_price >= donchian_high[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            elif close_price <= donchian_low[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Transition zone (20 <= ADX <= 25): hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Regime_Donchian_20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0