#!/usr/bin/env python3
"""
12h_1d_Stochastic_Bollinger_Trend_v1
Hypothesis: 12h StochRSI + Bollinger Bands with 1d ADX trend filter. 
StochRSI identifies overbought/oversold extremes within Bollinger Bands for mean reversion.
ADX filter ensures we only trade in trending markets to avoid whipsaws in ranging conditions.
Designed for low trade frequency (12-37/year) with clear entry/exit rules to minimize fee drag.
Works in both bull and bear markets by trading pullbacks in established trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Stochastic_Bollinger_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) on 12h close
    bb_length = 20
    bb_mult = 2.0
    basis = np.zeros(n)
    dev = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    # Simple moving average for basis
    for i in range(bb_length - 1, n):
        basis[i] = np.mean(close[i - bb_length + 1:i + 1])
    
    # Standard deviation for bands
    for i in range(bb_length - 1, n):
        dev[i] = np.std(close[i - bb_length + 1:i + 1])
    
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    
    # Calculate StochRSI (14, 14, 3, 3) on 12h close
    rsi_length = 14
    stoch_length = 14
    k_smooth = 3
    d_smooth = 3
    
    # RSI calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_length] = np.mean(gain[1:rsi_length + 1])
    avg_loss[rsi_length] = np.mean(loss[1:rsi_length + 1])
    
    for i in range(rsi_length + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (rsi_length - 1) + gain[i]) / rsi_length
        avg_loss[i] = (avg_loss[i - 1] * (rsi_length - 1) + loss[i]) / rsi_length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    stoch_rsi = np.zeros(n)
    for i in range(stoch_length - 1, n):
        min_rsi = np.min(rsi[i - stoch_length + 1:i + 1])
        max_rsi = np.max(rsi[i - stoch_length + 1:i + 1])
        if max_rsi - min_rsi != 0:
            stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
        else:
            stoch_rsi[i] = 50.0
    
    # Smooth K and D
    k = np.zeros(n)
    d = np.zeros(n)
    for i in range(k_smooth - 1, n):
        k[i] = np.mean(stoch_rsi[i - k_smooth + 1:i + 1])
    for i in range(d_smooth - 1, n):
        d[i] = np.mean(k[i - d_smooth + 1:i + 1])
    
    # Calculate ADX (14) on daily data for trend strength
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros(len(df_1d))
    dm_plus_smooth = np.zeros(len(df_1d))
    dm_minus_smooth = np.zeros(len(df_1d))
    
    # Initial values
    atr[tr_period] = np.mean(tr[:tr_period + 1])
    dm_plus_smooth[tr_period] = np.mean(dm_plus[:tr_period + 1])
    dm_minus_smooth[tr_period] = np.mean(dm_minus[:tr_period + 1])
    
    # Wilder's smoothing
    for i in range(tr_period + 1, len(df_1d)):
        atr[i] = (atr[i - 1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i - 1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i - 1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # Directional Indicators
    plus_di = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    minus_di = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros(len(df_1d))
    adx[tr_period * 2 - 1] = np.mean(dx[:tr_period * 2])
    
    for i in range(tr_period * 2, len(df_1d)):
        adx[i] = (adx[i - 1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(d[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending_market = adx_aligned[i] > 25
        
        # StochRSI signals: oversold < 20, overbought > 80
        stoch_oversold = d[i] < 20
        stoch_overbought = d[i] > 80
        
        # Bollinger Band position
        price_above_upper = close[i] > upper[i]
        price_below_lower = close[i] < lower[i]
        price_in_bands = lower[i] <= close[i] <= upper[i]
        
        # Entry conditions: StochRSI extreme + price at Bollinger Band + trend
        long_entry = stoch_oversold and price_below_lower and trending_market
        short_entry = stoch_overbought and price_above_upper and trending_market
        
        # Exit conditions: StochRSI returns to middle or trend weakens
        long_exit = (d[i] > 50 or adx_aligned[i] < 20)
        short_exit = (d[i] < 50 or adx_aligned[i] < 20)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals