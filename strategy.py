#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot (H3/L3) breakout with daily volume confirmation and 1-day ADX filter
# Long when price breaks above H3 pivot level AND daily volume > 1.3x 20-day average AND daily ADX > 25
# Short when price breaks below L3 pivot level AND daily volume > 1.3x 20-day average AND daily ADX > 25
# Exit when price returns to the 1-day close (pivot point) or reverses to opposite pivot level
# Camarilla levels provide institutional reference points, volume confirms institutional interest, ADX filters trending conditions
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for pivot levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous daily OHLC
    # H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    # Pivot point (close) used for exit
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    pivot_point = prev_close  # Camarilla pivot point = previous close
    
    # Align to 12h timeframe (wait for daily bar to close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    # Calculate daily volume average for confirmation (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate daily ADX for trend filter (14-period)
    # ADX requires +DI, -DI, and DX calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need 1 for shift + 20 for vol + 14 for ADX)
    start = 35
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(pivot_point_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg_1d_aligned[i] * 1.3
        
        if position == 0:
            # Long setup: price breaks above H3 + volume confirmation + ADX > 25
            if (price > camarilla_h3_aligned[i] and vol > vol_threshold and adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below L3 + volume confirmation + ADX > 25
            elif (price < camarilla_l3_aligned[i] and vol > vol_threshold and adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot point or breaks below L3 (reversal)
            if price <= pivot_point_aligned[i] or price < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot point or breaks above H3 (reversal)
            if price >= pivot_point_aligned[i] or price > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_H3L3_Volume_ADX"
timeframe = "12h"
leverage = 1.0