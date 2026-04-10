#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ADX(14) trend filter
# - Long when price breaks above Camarilla H3 level (1d) + ADX(14) > 25 + 1d volume > 1.3x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level (1d) + same ADX and volume conditions
# - Exit: opposite Camarilla level (H3 for shorts, L3 for longs) or close below/above L4/H4
# - Position sizing: 0.25 discrete level to balance risk and reward
# - Camarilla pivots from higher timeframe (1d) provide institutional support/resistance levels
# - ADX filter ensures we only trade when there's sufficient trend strength
# - Volume confirmation adds conviction to breakouts
# - Target: 20-50 trades/year on 4h timeframe to minimize fee drag

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    range_1d = high_1d - low_1d
    h3 = pivot + (range_1d * 1.1 / 2.0)  # H3 = pivot + 1.1*(HL)/2
    l3 = pivot - (range_1d * 1.1 / 2.0)  # L3 = pivot - 1.1*(HL)/2
    h4 = pivot + (range_1d * 1.1)        # H4 = pivot + 1.1*(HL)
    l4 = pivot - (range_1d * 1.1)        # L4 = pivot - 1.1*(HL)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 4h ADX for trend filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_period = 14
    atr = pd.Series(tr1).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, np.finfo(float).eps, atr)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / atr_safe
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero in DX calculation
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(30, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        
        # Volume confirmation: current 1d volume > 1.3x 20-period SMA (volume spike)
        vol_confirm = volume_1d_current[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Camarilla breakout signals
        long_breakout = close[i] > h3_aligned[i]
        short_breakout = close[i] < l3_aligned[i]
        
        # Exit conditions: opposite Camarilla level or extreme levels
        exit_long = close[i] < l3_aligned[i] or close[i] < l4_aligned[i]
        exit_short = close[i] > h3_aligned[i] or close[i] > h4_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if long_breakout and trending and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif short_breakout and trending and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals