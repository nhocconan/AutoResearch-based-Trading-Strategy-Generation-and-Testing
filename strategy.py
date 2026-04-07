#!/usr/bin/env python3
"""
4h_momentum_volume_regime_v1
Hypothesis: On 4h timeframe, enter long when RSI(14) crosses above 50 with volume above 20-period average and price above EMA(50), enter short when RSI(14) crosses below 50 with volume above average and price below EMA(50). Use 1d ADX(14) > 25 as trend filter to avoid range-bound whipsaws. Exit when RSI crosses back below 50 (long) or above 50 (short). Designed for 20-40 trades/year to minimize fee decay while capturing momentum shifts in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_momentum_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    if len(close) < 14:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    
    # Volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX(14) for trend regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trend_ok = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: RSI crosses back below 50
            if rsi[i] < 50 and rsi[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses back above 50
            if rsi[i] > 50 and rsi[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and trend_ok:
                # Long: RSI crosses above 50 with price above EMA50
                if rsi[i] > 50 and rsi[i-1] <= 50 and close[i] > ema_50[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI crosses below 50 with price below EMA50
                elif rsi[i] < 50 and rsi[i-1] >= 50 and close[i] < ema_50[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals