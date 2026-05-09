#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_DMI_Crossover_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    6h ADX + DMI crossover with 1d trend filter and volume confirmation.
    Long when +DI crosses above -DI with ADX>25 and price above 1d EMA(50).
    Short when -DI crosses above +DI with ADX>25 and price below 1d EMA(50).
    Exit when DMI cross reverses or ADX falls below 20.
    Volume filter: current volume > 1.8x 20-period average.
    Designed for 80-150 total trades over 4 years to balance signal quality and frequency.
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ADX and DMI (14-period)
    period = 14
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).values / atr
    
    # ADX calculation
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma20[i]
        
        if position == 0:
            # Long: +DI crosses above -DI with ADX>25, volume OK, and above 1d EMA
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and 
                adx[i] > 25 and vol_ok and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: -DI crosses above +DI with ADX>25, volume OK, and below 1d EMA
            elif (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and 
                  adx[i] > 25 and vol_ok and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: -DI crosses above +DI OR ADX falls below 20
            if (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: +DI crosses above -DI OR ADX falls below 20
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals