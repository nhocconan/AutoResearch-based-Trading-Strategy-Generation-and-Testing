#!/usr/bin/env python3
name = "1h_4h1d_Regime_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend and structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian(20) for breakout levels
    donchian_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d ADX(14) for trend strength regime
    # Calculate ADX components
    plus_dm = np.diff(df_4h['high'].values, prepend=df_4h['high'].values[0])
    minus_dm = np.diff(df_4h['low'].values, prepend=df_4h['low'].values[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(df_4h['high'].values - df_4h['low'].values)
    tr2 = np.abs(df_4h['high'].values - np.roll(df_4h['close'].values, 1))
    tr3 = np.abs(df_4h['low'].values - np.roll(df_4h['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:
            if trending:
                # Long: break above 4h Donchian high in uptrend with volume
                long_condition = (close[i] > donchian_high_aligned[i] and 
                                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                                volume[i] > vol_ma_24[i] * 1.5)
                # Short: break below 4h Donchian low in downtrend with volume
                short_condition = (close[i] < donchian_low_aligned[i] and 
                                 ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                                 volume[i] > vol_ma_24[i] * 1.5)
                
                if long_condition:
                    signals[i] = 0.20
                    position = 1
                elif short_condition:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit: price back below Donchian low or trend weakens
            if (close[i] < donchian_low_aligned[i] or 
                adx_aligned[i] < 20 or
                ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above Donchian high or trend weakens
            if (close[i] > donchian_high_aligned[i] or 
                adx_aligned[i] < 20 or
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h breakout of 4h Donchian channels with 1d ADX regime filter and volume confirmation
# - Uses 4h Donchian(20) breakouts for structural entry/exit
# - 1d ADX(25) filter ensures we only trade in trending markets (reduces whipsaws in ranging)
# - 1d EMA(50) provides trend direction for breakout bias
# - Volume confirmation (1.5x average) filters low-probability breakouts
# - Position size 0.20 manages risk while allowing meaningful returns
# - Designed to work in both bull and bear markets via trend filter
# - Target: 15-30 trades/year (60-120 over 4 years) to stay within fee limits
# - Exit when price returns to opposite Donchian level or trend weakens (ADX<20)