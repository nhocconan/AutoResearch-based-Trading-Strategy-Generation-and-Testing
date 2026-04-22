#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams %R Extreme Reversal with 1-day ADX trend filter and volume confirmation.
Trades reversals when Williams %R reaches oversold/overbought levels in the direction of daily trend.
Uses volume spike to confirm institutional interest. Designed for low trade frequency (20-50 trades/year)
to minimize fee drain and work in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator."""
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high, prepend=high[0]))
    tr2 = np.abs(np.diff(low, prepend=low[0]))
    tr3 = np.abs(np.diff(close, prepend=close[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values / (atr * period + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values / (atr * period + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Williams %R on 4h data (14-period)
    wr_14 = calculate_williams_r(high, low, close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(wr_14[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Williams %R oversold (< -80) with strong uptrend (ADX > 25)
            if wr_14[i] < -80 and adx_14_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with strong downtrend (ADX > 25)
            elif wr_14[i] > -20 and adx_14_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or trend weakens (ADX < 20)
            exit_signal = False
            
            if wr_14[i] > -50 or adx_14_1d_aligned[i] < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dADX25_Volume"
timeframe = "4h"
leverage = 1.0