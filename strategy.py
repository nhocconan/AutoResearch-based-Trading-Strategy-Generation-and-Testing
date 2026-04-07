#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d 200 EMA + 4h ADX + Volume Confirmation
# Hypothesis: 200 EMA defines long-term trend, ADX > 25 confirms strong trend,
# Volume > 1.5x average confirms institutional participation. Works in both bull/bear
# by only taking trades in direction of 200 EMA. Target: 10-25 trades/year (40-100 over 4 years).
name = "1d_200ema_4h_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 200 EMA on 1d timeframe
    close_s = pd.Series(close)
    ema_200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ADX on 4h timeframe (14 period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 4h indicators to 1d timeframe
    adx_1d = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema_200[i]) or np.isnan(adx_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 200 EMA or ADX drops below 20
            if close[i] < ema_200[i] or adx_1d[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above 200 EMA or ADX drops below 20
            if close[i] > ema_200[i] or adx_1d[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require ADX > 25 (strong trend) and volume confirmation
            if adx_1d[i] > 25 and vol_filter[i]:
                # Long: price above 200 EMA in uptrend
                if close[i] > ema_200[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below 200 EMA in downtrend
                elif close[i] < ema_200[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals