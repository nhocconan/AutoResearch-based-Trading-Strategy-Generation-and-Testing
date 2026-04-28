#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with 12h trend filter and volume confirmation
# ADX > 25 indicates strong trend, Alligator alignment (Lips > Teeth > Jaw for long, Lips < Teeth < Jaw for short) gives direction
# 12h EMA50 ensures higher-timeframe alignment, volume > 1.5x 20-bar average avoids chop
# Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Works in bull markets via trend continuation, in bear markets via strong trend shorts (2022 proved ADX strategies work in crashes)

name = "6h_ADX_Alligator_12hEMA50_Trend_Volume_v1"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator components (6h)
    # Jaw: SMA(13,8)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: SMA(8,5)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: SMA(5,3)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # ADX calculation (6h)
    # True Range
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high).diff()
    down_move = -pd.Series(low).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_vals = adx.values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 13, 8, 5, 14)  # EMA50, volume MA20, Alligator shifts, ADX period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(teeth_vals[i]) or np.isnan(jaw_vals[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(adx_vals[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        strong_trend = adx_vals[i] > 25
        price = close[i]
        lips_val = lips_vals[i]
        teeth_val = teeth_vals[i]
        jaw_val = jaw_vals[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Strong trend, bullish Alligator alignment, price above 12h EMA50, volume spike
            if strong_trend and lips_val > teeth_val and teeth_val > jaw_val and price > ema_50_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Strong trend, bearish Alligator alignment, price below 12h EMA50, volume spike
            elif strong_trend and lips_val < teeth_val and teeth_val < jaw_val and price < ema_50_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on trend weakening or stoploss
            # Exit on Alligator sleeping (crossing) or ADX weakening
            if lips_val <= teeth_val or teeth_val <= jaw_val or adx_vals[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on trend weakening or stoploss
            # Exit on Alligator sleeping (crossing) or ADX weakening
            if lips_val >= teeth_val or teeth_val >= jaw_val or adx_vals[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals