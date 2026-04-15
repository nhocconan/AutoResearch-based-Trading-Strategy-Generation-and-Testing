#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h volume-weighted breakout with 12h trend filter
# Uses 4h price crossing above/below 20-period VWAP as entry signal,
# confirmed by volume > 1.5x 20-period volume average and 12h ADX > 25.
# Exits when price crosses back below/above VWAP or ADX < 20 (ranging).
# Designed to capture trending moves with institutional volume confirmation,
# working in both bull and bear markets by following the 12h trend.
# Target: 80-150 total trades over 4 years (20-38/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 4h VWAP (20-period)
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume
    cum_vp = np.nancumsum(vp)
    cum_vol = np.nancumsum(volume)
    vwap = np.where(cum_vol > 0, cum_vp / cum_vol, typical_price)
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values with min_periods
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    dm_plus_series = pd.Series(dm_plus)
    dm_minus_series = pd.Series(dm_minus)
    dm_plus_smooth = dm_plus_series.rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = dm_minus_series.rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    dx_series = pd.Series(dx)
    adx = dx_series.rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 20-period volume average for confirmation
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after VWAP warmup
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price crosses above VWAP + volume confirmation + ADX > 25
        if (close[i] > vwap[i] and close[i-1] <= vwap[i-1] and
            volume[i] > 1.5 * vol_ma[i] and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price crosses below VWAP + volume confirmation + ADX > 25
        elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and
              volume[i] > 1.5 * vol_ma[i] and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price crosses back below/above VWAP or ADX < 20 (ranging)
        elif position == 1 and (close[i] < vwap[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > vwap[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_VWAP_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0