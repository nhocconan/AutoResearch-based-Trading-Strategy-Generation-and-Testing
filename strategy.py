#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + Volume Spike + ADX Trend Filter
# - Long when price breaks above Donchian upper band (20) + volume > 2x 20-period average + ADX > 25
# - Short when price breaks below Donchian lower band (20) + volume > 2x 20-period average + ADX > 25
# - Exit when price crosses back through Donchian middle band (20) or ADX drops below 20
# - Uses 1d ATR for volatility filter to avoid ranging markets
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (252-period ~ 1 year)
    atr_percentile = np.zeros_like(atr_1d)
    for i in range(252, len(atr_1d)):
        atr_percentile[i] = np.sum(atr_1d[i-252:i] <= atr_1d[i]) / 252 * 100
    
    # Align ATR percentile to 4h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate Donchian channels (20-period) on 4h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ADX(14) on 4h
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    up_14 = pd.Series(up_move).rolling(window=14, min_periods=14).mean().values
    down_14 = pd.Series(down_move).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * up_14 / tr_14
    minus_di = 100 * down_14 / tr_14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx[i]) or \
           np.isnan(atr_percentile_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: only trade when ATR percentile > 30 (avoid low volatility)
        vol_filter = atr_percentile_aligned[i] > 30
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + strong trend + volatility
            if (price > donchian_high[i] and 
                vol > 2.0 * vol_ma[i] and 
                adx[i] > 25 and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + strong trend + volatility
            elif (price < donchian_low[i] and 
                  vol > 2.0 * vol_ma[i] and 
                  adx[i] > 25 and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid or ADX weakens
            if price < donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid or ADX weakens
            if price > donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX_VolatilityFilter"
timeframe = "4h"
leverage = 1.0