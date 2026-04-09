#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ATR volume filter and 12h ADX regime filter
# - Long when price breaks above 20-period Donchian high + 12h volume > 1.5x 20-period average + 12h ADX > 25
# - Short when price breaks below 20-period Donchian low + 12h volume > 1.5x 20-period average + 12h ADX > 25
# - Uses ATR(14) trailing stop: exits when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Donchian breakouts capture trends, volume filter ensures participation, ADX filter avoids choppy markets
# - Works in both bull (trend continuation) and bear (trend reversals) markets

name = "4h_12h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h True Range for ATR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    
    # 12h ATR(14) for volatility and stoploss
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h Volume > 1.5x 20-period average (moderate filter for reasonable trade count)
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (1.5 * avg_volume_20)
    
    # 12h ADX(14) for regime filter
    # Calculate +DM and -DM
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth +DM, -DM, and TR
    tr_12h_smooth = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_12h_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_12h_smooth)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 12h indicators to 4h
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(volume_spike_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or atr_12h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and ADX confirmation
            if (high[i] >= donchian_high[i] and    # Break above Donchian high
                volume_spike_12h_aligned[i] and    # Volume confirmation
                adx_aligned[i] > 25):              # Strong trend regime
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= donchian_low[i] and    # Break below Donchian low
                  volume_spike_12h_aligned[i] and  # Volume confirmation
                  adx_aligned[i] > 25):            # Strong trend regime
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals