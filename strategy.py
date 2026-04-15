#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX-based trend following with 1d volatility filter
# Uses ADX to identify strong trends (trending when ADX>25, ranging when ADX<20),
# enters long when +DI crosses above -DI in trending markets with volatility filter,
# and short when -DI crosses above +DI. Volatility filter avoids entries during
# extreme volatility spikes that often lead to false breakouts. Designed to work
# in both bull and bear markets by only taking trades in the direction of the
# primary trend identified by ADX, with hysteresis to prevent whipsaw.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and ADX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 4h
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = np.diff(low_4h, prepend=low_4h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # ADX calculation
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR (14-period) on 1d for volatility filter
    tr1d = high_1d - low_1d
    tr2d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_d = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    tr_d[0] = tr1d[0]  # First value
    atr_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: avoid extreme volatility (ATR > 1.5 * 50-period MA of ATR)
    atr_ma_50d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_1d < 1.5 * atr_ma_50d
    
    # Align all indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_4h, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_4h, minus_di)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or
            np.isnan(minus_di_aligned[i]) or np.isnan(vol_filter_aligned[i])):
            continue
        
        # Long entry: ADX > 25 (trending) + +DI crosses above -DI + volatility filter
        if (adx_aligned[i] > 25 and
            plus_di_aligned[i] > minus_di_aligned[i] and
            plus_di_aligned[i-1] <= minus_di_aligned[i-1] and
            vol_filter_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: ADX > 25 (trending) + -DI crosses above +DI + volatility filter
        elif (adx_aligned[i] > 25 and
              minus_di_aligned[i] > plus_di_aligned[i] and
              minus_di_aligned[i-1] <= plus_di_aligned[i-1] and
              vol_filter_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: ADX < 20 (ranging market) or opposite DI crossover
        elif position == 1 and (adx_aligned[i] < 20 or minus_di_aligned[i] > plus_di_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_aligned[i] < 20 or plus_di_aligned[i] > minus_di_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_ADX_Trend_Filter_Volatility"
timeframe = "4h"
leverage = 1.0