#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R for mean reversion in bear/range markets combined with 1d ADX for regime filter.
# Long when Williams %R < -80 (oversold) and ADX < 25 (range/weak trend) - expects mean reversion bounce.
# Short when Williams %R > -20 (overbought) and ADX < 25 (range/weak trend) - expects mean reversion pullback.
# Uses discrete position size 0.25. Williams %R captures extreme price levels, ADX filter avoids whipsaws in strong trends.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    williams_r = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 13 or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or highest_high_14[i] == lowest_low_14[i]:
            williams_r[i] = -50.0  # neutral
        else:
            williams_r[i] = ((highest_high_14[i] - close_1d[i]) / (highest_high_14[i] - lowest_low_14[i])) * -100
    
    # === 1d Indicators: ADX (14-period) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    atr_14 = np.zeros_like(close_1d)
    plus_dm_14 = np.zeros_like(close_1d)
    minus_dm_14 = np.zeros_like(close_1d)
    
    # Initial values
    if len(close_1d) >= 14:
        atr_14[13] = np.sum(tr[1:15])
        plus_dm_14[13] = np.sum(plus_dm[1:15])
        minus_dm_14[13] = np.sum(minus_dm[1:15])
    
    # Wilder's smoothing: subsequent values
    for i in range(14, len(close_1d)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
        plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # Directional Indicators
    plus_di = np.where(atr_14 != 0, (plus_dm_14 / atr_14) * 100, 0)
    minus_di = np.where(atr_14 != 0, (minus_dm_14 / atr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros_like(close_1d)
    
    # Initial ADX value (first 14-period average of DX)
    if len(close_1d) >= 27:  # need 14 for initial DX + 14 for smoothing
        adx[26] = np.mean(dx[14:28])
    
    # Subsequent ADX values (Wilder's smoothing)
    for i in range(27, len(close_1d)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d Williams %R and ADX to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_val = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R rises above -50 (exiting oversold) or ADX > 30 (strong trend developing)
            if williams_val > -50 or adx_val > 30:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R falls below -50 (exiting overbought) or ADX > 30 (strong trend developing)
            if williams_val < -50 or adx_val > 30:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Regime filter: ADX < 25 (range/weak trend market)
            regime_filter = adx_val < 25
            
            # LONG: Williams %R < -80 (oversold) in range market
            if (williams_val < -80) and regime_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) in range market
            elif (williams_val > -20) and regime_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dWilliamsR_ADXRangeFilter_V1"
timeframe = "12h"
leverage = 1.0